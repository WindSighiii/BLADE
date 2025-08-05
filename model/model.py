import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from module import *
    
class BasicModel(nn.Module):
    '''
    put some public attributes and methods here...
    '''
    def __init__(self, config):
        super(BasicModel, self).__init__()

        self.config = config
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.behavior_types = config['behavior_types']
        self.cuda_condition = config['cuda_condition']
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.maxlen = config['maxlen']
        self.hidden_dims = config['hidden_dims']
        self.dropout_rate = config['dropout']
        self.init_std = config['init_std']
        
        self.hidden_dims = config['hidden_dims']
        self.dropout_rate = config['dropout']
        self.init_std = config['init_std']
        self.num_experts = config['num_experts']
        self.save_path = config['output_dir']
        self.num_blocks = config['num_blocks']
        self.num_heads = config['num_heads']
        self.no = config['no']
        self.use_mutl_gate=False
        self.save_path = config['output_dir']
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.no = config['no']
        self.model_name = str(self.__class__.__name__)
        fname = f"{self.model_name}_model.dataset={config['dataset']}." \
                f"hs={config['num_heads']}.es={config['num_experts']}.nb={config['num_blocks']}." \
                f"hd={config['hidden_dims']}.d={config['dropout']}.ml={config['maxlen']}." \
                f"rw={config['rec_weight']}.cw={config['cf_weight']}." \
                f"to={config['trade_off']}." \
                f"at={config['augment_type']}." \
                f"ep={config['max_epochs']}.st={config['seed']}.lr={config['lr']}." \
                f"no={self.no}.pth"
        self.save_path = os.path.join(self.save_path, fname)
        if isinstance(config['ffn_acti'], str):
            self.FFN_acti_fn = ACTIVATION_FUNCTION[config['ffn_acti']]
        else:
            self.FFN_acti_fn = config['ffn_acti']
    def init_weights(self, module):
        """
        Initialize the weights
        """
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        else:
            pass
class GatedFusion(nn.Module):
    def __init__(self, input_dim):
        super(GatedFusion, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 2) 
        )

    def forward(self, a, b):
        concat = torch.cat([a, b], dim=-1)
        scores = self.projection(concat) 
        weights = F.softmax(scores, dim=-1)
        feat1_weighted = a * weights[..., 0].unsqueeze(-1)  # [B, L, D]
        feat2_weighted = b * weights[..., 1].unsqueeze(-1)  # [B, L, D]
        output = feat1_weighted + feat2_weighted
        #output=
        return output
class GatedFusion1(nn.Module):
    def __init__(self, input_dim):
        super(GatedFusion1, self).__init__()
        self.gate_layer = nn.Linear(2 * input_dim, input_dim)

    def forward(self, a, b):
        concat = torch.cat([a, b], dim=-1)
        gate = torch.sigmoid(self.gate_layer(concat))
        output = gate * a + (1 - gate) * b
        #output=
        return output
class EarlyModel(nn.Module):
    def __init__(self,config):
        super(EarlyModel, self).__init__()
        self.fusion_type = config['early_fusion_type']
        self.hidden_dims = config['hidden_dims']
        if self.fusion_type == 'concat':
            self.fusion_layer = MLP(config)
        elif self.fusion_type == 'gate':
            self.fusion_layer = GatedFusion(self.hidden_dims)
        self.transformer = Transformer_early(config)
        self.emb_dropout=nn.Dropout(p=config['dropout'])
    def forward(self,u,item_emb,bs_emb,pos_emb,tbs_emb,attn_mask):
        if self.fusion_type == 'concat':
            hidden_states = torch.cat((item_emb,bs_emb),dim=-1)
            hidden_states=self.fusion_layer(hidden_states)
        elif self.fusion_type == 'gate':
            hidden_states=self.fusion_layer(item_emb,bs_emb)
        else:
            hidden_states=item_emb+bs_emb
        hidden_states=hidden_states+pos_emb
        hidden_states=self.emb_dropout(hidden_states)
        hidden_states=self.transformer(hidden_states,attn_mask)
        return hidden_states
    
class BLADE(BasicModel):
    def __init__(self,config):
        super(BLADE, self).__init__(config=config)
        self.trade_off = config['trade_off']
        # Add behavior prediction head
        self.emb_layernorm = nn.LayerNorm(self.hidden_dims, eps=1e-8)
        # embedding
        self.item_emb = nn.Embedding(self.item_num+1, self.hidden_dims, padding_idx=0)
        self.pos_emb = nn.Embedding(self.maxlen, self.hidden_dims,padding_idx=0)
        #self.pos_emb = nn.Embedding(self.maxlen, self.hidden_dims)
        self.behavior_emb=nn.Embedding(self.behavior_types+1, self.hidden_dims, padding_idx=0)
        self.user_factors=nn.Embedding(self.user_num+1, self.behavior_types, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=self.dropout_rate)
        #mask
        self.click_only_mask=None
        self.padding_mask = None
        self.attn_mask = None
        #early_fusion
        self.early_fusion = EarlyModel(config)
        #mid_fusion
        self.moe_encoder = MoeEncoder(config)
        self.cross_attention = CrossBlock(config)
        self.apply(self.init_weights)

    def set2emb(self, u, bs_seqs):
        batch_u_factors = self.user_factors(u.to(self.device)).unsqueeze(-2)
        bs_factors = nn.Softmax(dim=-1)(batch_u_factors * bs_seqs)
        behavior_set_embedding = torch.matmul(bs_factors, self.behavior_emb.weight[1:, :].unsqueeze(0))
        return behavior_set_embedding
    def set2emb_no_personalized(self, bs_seqs):
        
        behavior_set_embedding = torch.matmul(bs_seqs,
                                      self.behavior_emb.weight[1:, :].unsqueeze(0))
        return behavior_set_embedding
    
    def seqsEncoding(self,input_seq):
        position_ids = torch.arange(self.maxlen, dtype=torch.long,
                                    device=input_seq.device).unsqueeze(0).expand_as(input_seq)
        item_embeddings = self.item_emb(input_seq)
        item_embeddings *= self.item_emb.embedding_dim ** 0.5
        position_embeddings = self.pos_emb(position_ids)
        #item_embeddings=self.frequency_layer(item_embeddings,self.click_only_mask)
        seq_emb = item_embeddings + position_embeddings
        seq_emb = self.emb_dropout(seq_emb)
        # (b, L, d)
        return seq_emb,item_embeddings,position_embeddings
    
    def _generate_mask(self, input_ids):
        # construct attention mask
        # (b, L)
        padding_mask = (input_ids > 0).long()
        origin_padding_mask = padding_mask.clone()

        # (b, L) --> (b, 1, 1, L)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        attn_mask_shape = (1, self.maxlen, self.maxlen)  # (1, L, L)
        # upper triangular, torch.uint8
        attn_mask = torch.triu(torch.ones(attn_mask_shape), diagonal=1)
        # --> bool
        attn_mask = (attn_mask == 0).unsqueeze(1)
        attn_mask = attn_mask.long()

        if self.config['cuda_condition']:
            attn_mask = attn_mask.cuda()

        # (b, 1, 1, L) * (1, 1, L, L)  --> (b, 1, L, L)
        attn_mask = padding_mask * attn_mask
        # (b, 1, L, L)
        attn_mask = (1.0 - attn_mask) * -1e5

        return origin_padding_mask, attn_mask
    
    def _early_forward(self, u, item_emb, pos_emb,bs_emb,target_bs_emb):

        hidden_states=self.early_fusion(u,item_emb,bs_emb,pos_emb,target_bs_emb,self.attn_mask)
        return hidden_states
    def _mid_forward(self, u, seqs_emb, bs_emb,target_bs_emb):
        '''
        b: batch size;
        L: maxlen of seq;
        bt: behavior types num
        
        :param u: (b,)
        :param input_seqs: (b, L)
        :param bs_seqs: (b, L, bt)
        :return: output: (b, L, d)
        '''
 
        X_emp=self.moe_encoder(seqs_emb,bs_emb,self.attn_mask)

        return X_emp
    
    def forward(self,users,item_seqs,behaviors_seqs,target_behaviors):
        self.padding_mask, self.attn_mask = self._generate_mask(item_seqs)
        self.click_only_mask=computeMask(behaviors_seqs)

        seqs_emb,item_emb,pos_emb=self.seqsEncoding(item_seqs)
        bs_emb=self.set2emb(users,behaviors_seqs)
        target_bs_emb=self.set2emb(users,target_behaviors)
        

  
        X_mid=self._mid_forward(users,seqs_emb,bs_emb,target_bs_emb)
        X_early=self._early_forward(users,item_emb,pos_emb,bs_emb,target_bs_emb)
        X=X_mid*self.trade_off+X_early*(1-self.trade_off)
        X=self.cross_attention(target_bs_emb,X,X,self.attn_mask)
        return X


