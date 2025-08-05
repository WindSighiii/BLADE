import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import squareform, pdist
import torch.fft
import faiss
def gelu(x):
    """
    Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415

    Reference from: ICLRec: https://github.com/salesforce/ICLRec
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def elu(x, alpha=1.0):
    return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))

ACTIVATION_FUNCTION = {
    "gelu": gelu,
    "relu": F.relu,
    "elu": elu
}
class LayerNorm(nn.Module):
    """
    Construct a layernorm module in the TF style (epsilon inside the square root).
    Reference from: ICLRec: https://github.com/salesforce/ICLRec
    """
    def __init__(self, hidden_size, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
class LayerNorm1(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Reference: https://arxiv.org/abs/1910.07467
    Like LayerNorm, but only uses the variance (no centering).
    """
    def __init__(self, hidden_size, eps=1e-8):
        super(LayerNorm1, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # Compute variance only (no mean subtraction)
        norm = x.norm(dim=-1, keepdim=True)  # L2 norm
        rms = norm / (x.shape[-1] ** 0.5)  # Root Mean Square
        x = x / (rms + self.eps)
        return self.weight * x
class BatchNorm(nn.Module):
    '''
    take the input as the shape of (B, L, D), different from nn.BatchNorm1d (N, C, L)
    Reference from: https://github.com/Antinomy20001/BatchNorm_Pytorch_Experiment/blob/master/BatchNorm.py
    '''
    def __init__(self, hidden_dims):
        super(BatchNorm, self).__init__()
        self.hidden_dims = hidden_dims
        self.eps = 1e-5
        self.momentum = 0.1

        # hyper parameters
        self.gamma = nn.Parameter(torch.Tensor(self.hidden_dims), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(self.hidden_dims), requires_grad=True)

        # moving average
        self.moving_mean = torch.zeros(self.hidden_dims)
        self.moving_var = torch.ones(self.hidden_dims)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.moving_var)
        nn.init.zeros_(self.moving_mean)

    def batch_norm(self, x, gamma, beta, moving_mean, moving_var,
                   is_training=True, eps=1e-5, momentum=0.9):
        assert x.shape[-1] == self.hidden_dims
        mu = torch.mean(x, dim=(0, 1), keepdim=True) # (d, )
        var = torch.std(x, dim=(0, 1), unbiased=False) # (d, )
        if is_training:
            x_hat = (x - mu) / torch.sqrt(var + eps)
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mu
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        else:
            x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
        out = gamma * x_hat + beta

        return out, moving_mean, moving_var

    def forward(self, x):
        '''
        :param x: expected (b, L, d)
        '''
        self.moving_mean = self.moving_mean.to(x.device)
        self.moving_var = self.moving_var.to(x.device)

        bn_x, self.moving_mean, self.moving_var = self.batch_norm(x, self.gamma, self.beta,
                                                                  self.moving_mean, self.moving_var,
                                                                  self.training, self.eps, self.momentum)
        return bn_x
    
class NCELoss(nn.Module):

    def __init__(self, config):
        super(NCELoss, self).__init__()
        self.device = torch.device("cuda" if config['cuda_condition'] else "cpu")
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = config['temperature']
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    def forward(self, batch_sample_one, batch_sample_two, intent_ids=None):
        batch_sample_one = F.normalize(batch_sample_one, p=2, dim=-1)
        batch_sample_two = F.normalize(batch_sample_two, p=2, dim=-1)
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        if intent_ids is not None:
            intent_ids = intent_ids.contiguous().view(-1, 1)
            mask_11_22 = torch.eq(intent_ids, intent_ids.T).long().to(self.device)
            sim11[mask_11_22 == 1] = float("-inf")
            sim22[mask_11_22 == 1] = float("-inf")
            eye_metrix = torch.eye(d, dtype=torch.long).to(self.device)
            mask_11_22[eye_metrix == 1] = 0
            sim12[mask_11_22 == 1] = float("-inf")
        else:
            mask = torch.eye(d, dtype=torch.long).to(self.device)
            INF = -1e9  # safe negative mask
            #sim11[mask == 1] = float("-inf")
            #sim22[mask == 1] = float("-inf")
            sim11[mask == 1] = INF
            sim22[mask == 1] = INF
        
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss
    


class PCLoss(nn.Module):

    def __init__(self, config, contrast_mode="all"):
        super(PCLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.criterion = NCELoss(config)

    def forward(self, batch_sample_one, batch_sample_two, intents, intent_ids):
        """
        features:
        intents: num_clusters x batch_size x hidden_dims
        """
        mean_pcl_loss = 0
        if intent_ids is not None:
            for intent, intent_id in zip(intents, intent_ids):
                pos_one_compare_loss = self.criterion(batch_sample_one, intent, intent_id)
                pos_two_compare_loss = self.criterion(batch_sample_two, intent, intent_id)
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss

            mean_pcl_loss /= 2 * len(intents)
        else:
            for intent in intents:
                pos_one_compare_loss = self.criterion(batch_sample_one, intent, intent_ids=None)
                pos_two_compare_loss = self.criterion(batch_sample_two, intent, intent_ids=None)
                mean_pcl_loss += pos_one_compare_loss

                mean_pcl_loss += pos_two_compare_loss

            mean_pcl_loss /= 2 * len(intents)
        return mean_pcl_loss

    
class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.num_heads = config['num_heads']
        self.hidden_dims = config['hidden_dims']
        if self.hidden_dims % self.num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_dims, self.num_heads)
            )
        self.head_size = int(self.hidden_dims / self.num_heads)
        self.reconstruct_size = self.num_heads * self.head_size

        self.Q = nn.Linear(self.hidden_dims, self.reconstruct_size)
        self.K = nn.Linear(self.hidden_dims, self.reconstruct_size)
        self.V = nn.Linear(self.hidden_dims, self.reconstruct_size)

        self.FCL = nn.Linear(self.reconstruct_size, self.hidden_dims)
        self.Q_layer_norm = LayerNorm(self.hidden_dims)
        self.final_layer_norm = LayerNorm(self.hidden_dims)

    def split_head(self, input):
        split_tensor_shape = input.size()[:-1] + (self.num_heads, self.head_size)
        input = input.view(*split_tensor_shape)
        # (b, h, L, d/h)
        return input.permute(0, 2, 1, 3)

    def forward(self, inputs, mask):
        raise NotImplementedError
class middle_attention(SelfAttention):
    def __init__(self, config):
        super(middle_attention, self).__init__(config)
        self.attn_dropout_layer = nn.Dropout(config['dropout'])
        self.out_dropout_layer = nn.Dropout(config['dropout'])
        self.head_size = int(self.hidden_dims / self.num_heads)
        self.Q_behavior = nn.Linear(config['hidden_dims'], config['hidden_dims'])
        self.K_behavior = nn.Linear(config['hidden_dims'], config['hidden_dims'])
        self.V_behavior = nn.Linear(config['hidden_dims'], config['hidden_dims'])
    def forward(self, item_embeds, behavior_embeds, attn_mask):
            # inputs: (batch, seq_len, hidden_dims)
        batch_size, seq_len, _ = item_embeds.size()

        Q = self.Q(item_embeds)  # (batch, seq_len, reconstruct_size)
        K = self.K(item_embeds)
        V = self.V(item_embeds)
        Q_behavior = self.Q_behavior(behavior_embeds)
        K_behavior = self.K_behavior(behavior_embeds)
        V_behavior = self.V_behavior(behavior_embeds)

        Q = self.split_head(Q)  # (batch,num_heads,seq_len,head_size)
        K = self.split_head(K)
        V = self.split_head(V)
        Q_behavior = self.split_head(Q_behavior)
        K_behavior = self.split_head(K_behavior)
        V_behavior = self.split_head(V_behavior)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, num_heads, seq_len, seq_len)
        attn_scores_behavior = torch.matmul(Q_behavior, K_behavior.transpose(-2, -1))
        attn_scores = (attn_scores+attn_scores_behavior) / math.sqrt(self.head_size)


        if attn_mask is not None:
            # mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
            attn_scores = attn_scores + attn_mask


        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout_layer(attn_probs)

        attn_output = torch.matmul(attn_probs, V)  # (batch, num_heads, seq_len, head_size)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, num_heads, head_size)
        attn_output = attn_output.view(batch_size, seq_len, -1)  # (batch, seq_len, reconstruct_size)

        output = self.FCL(attn_output)
        output = self.out_dropout_layer(output)
        output = self.final_layer_norm(output + item_embeds)
        return output
class MultiHeadAttention_early(SelfAttention):
    def __init__(self, config):
        super(MultiHeadAttention_early, self).__init__(config)
        self.num_heads = config['transformer_num_heads']
        self.attn_dropout_layer = nn.Dropout(config['transformer_dropout'])
        self.head_size = int(self.hidden_dims / self.num_heads)
    def forward(self, inputs, attn_mask):
            # inputs: (batch, seq_len, hidden_dims)
        batch_size, seq_len, _ = inputs.size()

        Q = self.Q(inputs)  # (batch, seq_len, reconstruct_size)
        K = self.K(inputs)
        V = self.V(inputs)

        Q = self.split_head(Q)  # (batch,num_heads,seq_len,head_size)
        K = self.split_head(K)
        V = self.split_head(V)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, num_heads, seq_len, seq_len)
        attn_scores = attn_scores / math.sqrt(self.head_size)

        if attn_mask is not None:
            # mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
            attn_scores = attn_scores + attn_mask


        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout_layer(attn_probs)

        attn_output = torch.matmul(attn_probs, V)  # (batch, num_heads, seq_len, head_size)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, num_heads, head_size)
        attn_output = attn_output.view(batch_size, seq_len, -1)  # (batch, seq_len, reconstruct_size)

        output = self.FCL(attn_output)
        output = self.final_layer_norm(output + inputs)

        return output
class MultiHeadAttention(SelfAttention):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__(config)
        self.num_heads = config['num_heads']
        self.attn_dropout_layer = nn.Dropout(config['dropout'])
        self.head_size = int(self.hidden_dims / self.num_heads)
    def forward(self, inputs, attn_mask):
            # inputs: (batch, seq_len, hidden_dims)
        batch_size, seq_len, _ = inputs.size()

        Q = self.Q(inputs)  # (batch, seq_len, reconstruct_size)
        K = self.K(inputs)
        V = self.V(inputs)

        Q = self.split_head(Q)  # (batch,num_heads,seq_len,head_size)
        K = self.split_head(K)
        V = self.split_head(V)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, num_heads, seq_len, seq_len)
        attn_scores = attn_scores / math.sqrt(self.head_size)

        if attn_mask is not None:
            # mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
            attn_scores = attn_scores + attn_mask


        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout_layer(attn_probs)

        attn_output = torch.matmul(attn_probs, V)  # (batch, num_heads, seq_len, head_size)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, num_heads, head_size)
        attn_output = attn_output.view(batch_size, seq_len, -1)  # (batch, seq_len, reconstruct_size)

        output = self.FCL(attn_output)
        output = self.final_layer_norm(output + inputs)

        return output
class CrossAttention(SelfAttention):
    def __init__(self, config):
        super(CrossAttention, self).__init__(config)
        self.attn_dropout_layer = nn.Dropout(config['dropout'])
    def forward(self, query,key,value, attn_mask):
            # inputs: (batch, seq_len, hidden_dims)
        batch_size, seq_len, _ = query.size()

        Q = self.Q(query)  # (batch, seq_len, reconstruct_size)
        K = self.K(key)
        V = self.V(value)

        Q = self.split_head(Q)  # (batch, seq_len, num_heads, head_size)
        K = self.split_head(K)
        V = self.split_head(V)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, num_heads, seq_len, seq_len)
        attn_scores = attn_scores / math.sqrt(self.head_size)

        if attn_mask is not None:
            # mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
            attn_scores = attn_scores + attn_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout_layer(attn_probs)

        attn_output = torch.matmul(attn_probs, V)  # (batch, num_heads, seq_len, head_size)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, num_heads, head_size)
        attn_output = attn_output.view(batch_size, seq_len, -1)  # (batch, seq_len, reconstruct_size)

        output = self.FCL(attn_output)
        output = self.final_layer_norm(output + query)

        return output
class CrossBlock(nn.Module):
    def __init__(self, config):
        super(CrossBlock, self).__init__()
        self.cross = CrossAttention(config)
        self.ffn = FFNBlock(config)
    def forward(self, query, key, value, attention_mask):
        outputs = self.cross(query, key, value, attention_mask)
        outputs = self.ffn(outputs)
        return outputs   
class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_dims = config['hidden_dims']
        self.conv1 = nn.Conv1d(self.hidden_dims, self.hidden_dims, kernel_size=(1,))
        self.dropout1 = nn.Dropout(config['dropout'])
        if isinstance(config['ffn_acti'], str):
            self.FFN_acti_fn = ACTIVATION_FUNCTION[config['ffn_acti']]
        else:
            self.FFN_acti_fn = config['ffn_acti']
        self.conv2 = nn.Conv1d(self.hidden_dims, self.hidden_dims, kernel_size=(1,))
        self.dropout2 = nn.Dropout(config['dropout'])
    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.FFN_acti_fn(
            self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class MoeEncoder(nn.Module):
    def __init__(self,config):
        super(MoeEncoder,self).__init__()
        self.hidden_dims = config['hidden_dims']
        self.num_experts = config['num_experts']
        self.num_blocks = config['num_blocks']
        self.seqs_blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            seqs_block = MoeBlock(config)
            self.seqs_blocks.append(seqs_block)
    def forward(self,seqs,behaviors,attention_mask):
        for seqs_block in self.seqs_blocks:
            seqs = seqs_block(seqs,behaviors,attention_mask)
        return seqs

class MoeEncoder_middle(nn.Module):
    def __init__(self,config):
        super(MoeEncoder_middle,self).__init__()
        self.hidden_dims = config['hidden_dims']
        self.num_experts = config['num_experts']
        self.num_blocks = config['num_blocks']
        self.seqs_blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            seqs_block = middle_moe_block(config)
            self.seqs_blocks.append(seqs_block)
    def forward(self,seqs,behaviors,attention_mask):
        for seqs_block in self.seqs_blocks:
            seqs = seqs_block(seqs,behaviors,attention_mask)
        return seqs
    
class TransformerBlock_early(nn.Module):
    def __init__(self,config):
        super(TransformerBlock_early,self).__init__()
        self.hidden_dims = config['hidden_dims']
        self.attn = MultiHeadAttention_early(config)
        self.ffn = FFNBlock(config)
        self.ln1 = LayerNorm(config['hidden_dims'])
    def forward(self,seqs,attention_mask):
        outputs = self.attn(seqs,attention_mask)

        outputs=self.ln1(seqs+outputs)######
        outputs = self.ffn(outputs)
        return outputs
class Transformer_early(nn.Module):
    def __init__(self,config):
        super(Transformer_early,self).__init__()
        self.num_blocks = config['transformer_num_blocks']
        self.seqs_blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            seqs_block = TransformerBlock_early(config)
            self.seqs_blocks.append(seqs_block)
    def forward(self,seqs,attention_mask):
        for seqs_block in self.seqs_blocks:
            seqs = seqs_block(seqs,attention_mask)
        return seqs
    
class TransformerBlock(nn.Module):
    def __init__(self,config):
        super(TransformerBlock,self).__init__()
        self.hidden_dims = config['hidden_dims']
        self.num_heads = config['num_heads']
        self.attn = MultiHeadAttention(config)
        self.ffn = FFNBlock(config)
    def forward(self,seqs,attention_mask):
        outputs = self.attn(seqs,attention_mask)
        outputs = self.ffn(outputs)
        return outputs
class Transformer(nn.Module):
    def __init__(self,config):
        super(Transformer,self).__init__()
        self.num_blocks = config['num_blocks']
        self.seqs_blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            seqs_block = TransformerBlock(config)
            self.seqs_blocks.append(seqs_block)
    def forward(self,seqs,attention_mask):
        for seqs_block in self.seqs_blocks:
            seqs = seqs_block(seqs,attention_mask)
        return seqs
class FFNBlock(nn.Module):
    def __init__(self, config):
        super(FFNBlock, self).__init__()
        self.hidden_dims = config['hidden_dims']
        self.ffn = FeedForwardNetwork(config)
        self.layernorm = LayerNorm(self.hidden_dims)
    def forward(self, inputs):
        outputs = self.ffn(inputs)
        outputs = self.layernorm(outputs)
        return outputs

class Moe(torch.nn.Module):
    def __init__(self, config):
        super(Moe, self).__init__()
        self.hidden_dims = config['hidden_dims']
        self.num_experts = config['num_experts']
        self.dropout_rate = config['dropout']
        self.gate = torch.nn.Embedding(self.hidden_dims, self.num_experts)
        
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.experts= torch.nn.ModuleList()
        for _ in range(self.num_experts):
            ffn=FeedForwardNetwork(config)
            self.experts.append(ffn)           
        
        
    def forward(self, inputs,behaviors):
        gate_scores = torch.matmul(behaviors, self.gate.weight)
        gate_scores = F.softmax(gate_scores, dim=-1)
            
        expert_outputs = [expert(inputs) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)
            
        outputs = torch.sum(expert_outputs * gate_scores.unsqueeze(-2), dim=-1)
        return outputs

class MoeBlock(torch.nn.Module):
    def __init__(self, config):
        super(MoeBlock, self).__init__()
        self.attn =MultiHeadAttention(config)
        self.moe= Moe(config)
        self.ln1 = nn.LayerNorm(config['hidden_dims'], eps=1e-8)##?
        self.ln2 = nn.LayerNorm(config['hidden_dims'], eps=1e-8)
    def forward(self,seqs,behaviors,attention_mask):
        mha_outputs= self.attn(seqs,attn_mask=attention_mask)
        x= self.ln1(seqs + mha_outputs)##?
        seqs = self.moe(x,behaviors)
        seqs = self.ln2(x+seqs)
        return seqs

class middle_moe_block(nn.Module):
    def __init__(self, config):
        super(middle_moe_block, self).__init__()
        self.middle_attn = middle_attention(config)
        self.moe= Moe(config)
        self.ln1 = nn.LayerNorm(config['hidden_dims'], eps=1e-8)
        self.ln2 = nn.LayerNorm(config['hidden_dims'], eps=1e-8)
    def forward(self,seqs,behaviors,attention_mask):
        middle_attn_outputs= self.middle_attn(seqs,behaviors,attn_mask=attention_mask)
        x= self.ln1(seqs + middle_attn_outputs)
        seqs = self.moe(x,behaviors)
        seqs = self.ln2(x+seqs)
        return seqs
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.input_dim = config['hidden_dims']*2
        self.hidden_dim = config['hidden_dims']
        if isinstance(config['ffn_acti'], str):
            self.FFN_acti_fn = ACTIVATION_FUNCTION[config['ffn_acti']]
        else:
            self.FFN_acti_fn = config['ffn_acti']
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
    def forward(self,inputs):
        x = self.fc1(inputs)
        x = self.FFN_acti_fn(x)
        x = self.fc2(x)
        return x

