from tqdm import tqdm
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import torch.nn.functional as F
from module import NCELoss,PCLoss
class Trainer:
    def __init__(self, model, train_dataloader,valid_dataloader, test_dataloader, config):
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.behavior_types = config['behavior_types']
        self.hidden_dims = config['hidden_dims']

        self.config = config
        self.cuda_condition = config['cuda_condition']
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model
        if self.cuda_condition:
            self.model.cuda()
        self.output_path = model.save_path

        self.max_epochs = config['max_epochs']
        self.log_freq = config['log_freq']
        self.eval_freq = config['eval_freq']
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        betas = (config['adam_beta1'], config['adam_beta2'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'],
                                          betas=betas, weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    factor=config['decay_factor'],
                                                                    verbose=True,
                                                                    min_lr=config['min_lr'],
                                                                    patience=config['patience'])

        self.eval_item_mask = self.generate_eval_item_mask()
        self.test_item_mask = self.generate_test_item_mask()
        self.print_out_epoch = 0
        self.cf_criterion = NCELoss(self.config)
        self.pcl_criterion = PCLoss(self.config)
        
    def _instance_cl_one_pair_contrastive_learning(self, inputs,intent_ids=None):
        id_tensors,aug_tensors = inputs 
        (uid, input_item, input_bs,
                     target_item, target_bs, target_neg, ground_truth) = id_tensors
        repead_uid=torch.cat([uid,uid],dim=0)
        repead_items=torch.cat([input_item,input_item],dim=0)
        aug_tensors_1,aug_tensors_2=aug_tensors
        aug_cur_1,aug_tar_1=aug_tensors_1
        aug_cur_2,aug_tar_2=aug_tensors_2
        concat_target_bs=torch.cat([aug_tar_1,aug_tar_2],dim=0)
        concat_cur_bs = torch.cat([aug_cur_1,aug_cur_2], dim=0)

        cl_sequence_output = self.model(repead_uid,repead_items,concat_cur_bs,concat_target_bs)

        if self.config['seq_representation_instancecl_type'] == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.reshape(concat_cur_bs.shape[0], -1)
        batch_size = concat_cur_bs.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)

        if self.config['de_noise']:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids)
        else:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None)
        return cl_loss
    




    def generate_eval_item_mask(self):
        row = []
        col = []
        data = []
        for _, id_tensors in enumerate(self.valid_dataloader):
            uid, input_item = id_tensors[0], id_tensors[1]
            uid = uid.numpy()
            input_item = input_item.numpy()
            for idx, u in enumerate(uid):
                # for padding idx 0
                row.append(u)
                col.append(0)
                data.append(1)
                for i in input_item[idx]:
                    row.append(u)
                    col.append(i)
                    data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        eval_item_mask = csr_matrix((data, (row, col)),
                                    shape=(self.user_num + 1, self.item_num + 1))

        return eval_item_mask

    def generate_test_item_mask(self):
        row = []
        col = []
        data = []
        for _, id_tensors in enumerate(self.test_dataloader):
            uid, input_item = id_tensors[0], id_tensors[1]
            uid = uid.numpy()
            input_item = input_item.numpy()
            for idx, u in enumerate(uid):
                # for padding idx 0
                row.append(u)
                col.append(0)
                data.append(1)
                for i in input_item[idx]:
                    row.append(u)
                    col.append(i)
                    data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        test_item_mask = csr_matrix((data, (row, col)),
                                    shape=(self.user_num + 1, self.item_num + 1))

        return test_item_mask

    def train(self):
        self.fit(self.train_dataloader)

    def valid(self):
        return self.fit(self.valid_dataloader, mode="eval")

    def test(self):
        return self.fit(self.test_dataloader, mode="test")

    def fit(self, dataloader, mode="train"):
        raise NotImplementedError

    def load(self):
        self.model.load_state_dict(torch.load(self.output_path))

    def save(self):
        torch.save(self.model.cpu().state_dict(), self.output_path)
        self.model.to(self.device)
    
    def BCELoss(self, seq_output, pos_ids, neg_ids, target_bs=None):

        '''
        Binary Cross Entropy Loss with behavior richness weighting
        Args:
            seq_output: sequence output from model (b, L, d)
            pos_ids: positive item ids (b, L)
            neg_ids: negative item ids (b, L)
            target_bs: target behaviors (b, L, bt) - if provided, will use for behavior richness weighting
        '''
        # (b, L, d)
        pos_emb = self.model.item_emb(pos_ids)
        neg_emb = self.model.item_emb(neg_ids)

        # (bL, d)
        D = pos_emb.size(2)
        pos_item_emb = pos_emb.view(-1, D)
        neg_item_emb = neg_emb.view(-1, D)

        # (bL, d)
        x = seq_output.contiguous().view(-1, D)

        # (bL, )
        pos_item_logits = torch.sum(pos_item_emb * x, -1)
        neg_item_logits = torch.sum(neg_item_emb * x, -1)

        # (bL, )
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.maxlen).float()

        # Calculate behavior richness weights if target_bs is provided
        if target_bs is not None:
            # (bL, bt)
            target_bs_reshaped = target_bs.view(-1, target_bs.size(-1))
            # Calculate number of behaviors per item (bL,)
            behavior_count = torch.sum(target_bs_reshaped, dim=-1)
            # Normalize behavior count to get weights (bL,)
            behavior_weights = (behavior_count / self.behavior_types) * istarget
            # Add small constant to avoid zero weights
            behavior_weights = behavior_weights + 1e-24
        else:
            behavior_weights = istarget

        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_item_logits) + 1e-24) * behavior_weights
            - torch.log(1 - torch.sigmoid(neg_item_logits) + 1e-24) * behavior_weights
        ) / torch.sum(behavior_weights)

        return loss


    def calculate_all_item_prob(self, output):
        # (|I|+1, d)
        item_emb_weight = self.model.item_emb.weight
        # item side prob
        # (b, d) * (d, |I|+1) --> (b, |I|+1)
        item_prob = torch.matmul(output, item_emb_weight.transpose(0, 1))
        return item_prob

    def calculate_eval_metrics(self, mode, print_out_epoch, pred_item_list, ground_truth_list):
        NDCG_n_list, HR_n_list = [], []
        for k in [5, 10, 20]:
            NDCG_n_list.append(ndcg_k(pred_item_list, ground_truth_list, k))
            HR_n_list.append(hr_k(pred_item_list, ground_truth_list, k))

        eval_metrics_info = {
            "Epoch": print_out_epoch,
            "HR@5": "{:.4f}".format(HR_n_list[0]),
            "NDCG@5": "{:.4f}".format(NDCG_n_list[0]),
            "HR@10": "{:.4f}".format(HR_n_list[1]),
            "NDCG@10": "{:.4f}".format(NDCG_n_list[1]),
            "HR@20": "{:.4f}".format(HR_n_list[2]),
            "NDCG@20": "{:.4f}".format(NDCG_n_list[2]),
        }

        if mode == 'eval':
            print(set_color(str(eval_metrics_info), "cyan"))
        return [HR_n_list[0], NDCG_n_list[0], HR_n_list[1],
                NDCG_n_list[1], HR_n_list[2], NDCG_n_list[2]], str(eval_metrics_info)



class BLADETrainer(Trainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, config):
        super(BLADETrainer, self).__init__(
            model, train_dataloader, valid_dataloader, test_dataloader, config
        )
        self.loss_history = {
            'bce_loss': [],
            'cl_loss': [],
            'joint_loss': [],
            'epochs': []
        }
    def fit(self, dataloader,mode="train"):
        assert mode in {"train", "eval", "test"}
        print(set_color("Rec Model mode: " + mode, "green"))

        if mode == "train":
            self.model.train()

            early_stopping = EarlyStopping(save_path=self.output_path)
            print(set_color(f"Rec dataset Num of batch: {len(dataloader)}", "white"))
            # 训练阶段
            for epoch in range(self.max_epochs):
                if self.iteration(dataloader,epoch,early_stopping):
                    break

            if not early_stopping.early_stop:
                print("Reach the max number of epochs!")
                best_scores_info = {
                    "HR@5": "{:.4f}".format(early_stopping.best_scores[0]),
                    "NDCG@5": "{:.4f}".format(early_stopping.best_scores[1]),
                    "HR@10": "{:.4f}".format(early_stopping.best_scores[2]),
                    "NDCG@10": "{:.4f}".format(early_stopping.best_scores[3]),
                    "HR@20": "{:.4f}".format(early_stopping.best_scores[4]),
                    "NDCG@20": "{:.4f}".format(early_stopping.best_scores[5]),
                }

                print(set_color(f'\nBest Valid (' +
                                str(early_stopping.best_valid_epoch) +
                                ') Scores: ' +
                                str(best_scores_info) + '\n', 'cyan'))

            # test phase
            self.model.load_state_dict(torch.load(self.output_path))
            _, test_info = self.test()
            print(set_color(f'\nFinal Test Metrics: ' +
                            test_info + '\n', 'pink'))
            with open('./eval_metrics_log.txt', 'a') as f:
                f.write(str(self.config['frequency_dropout'])+'-------:')
                f.write(test_info + '\n')
            

        else:
            item_mask = self.eval_item_mask
            if mode == "test":
                item_mask = self.test_item_mask

            self.model.eval()
            iter_data = tqdm(enumerate(dataloader), total=len(dataloader))
            pred_item_list = None
            ground_truth_list = None
            target_bs_list= None 
            uid_list=None
            with torch.no_grad():
                for i, id_tensors in iter_data:
                    id_tensors = tuple(t.to(self.device) for t in id_tensors)
                    (uid, input_item, input_bs,
                     target_item, target_bs, target_neg, ground_truth) = id_tensors
                    seq_output= self.model(uid, input_item, input_bs, target_bs)
                    seq_output = seq_output[:, -1, :]
                    # batch of recommendation results
                    # (b, |I|+1)
                    item_prob = self.calculate_all_item_prob(seq_output).cpu().data.numpy().copy()
                    batch_user_idx = uid.cpu().numpy()
                    item_prob[item_mask[batch_user_idx].toarray() > 0] = -np.inf # (b, |I|+1)

                    # extract top-20 prob of item idx
                    top_idx = np.argpartition(item_prob, -20)[:, -20:]
                    topn_prob = item_prob[np.arange(len(top_idx))[:, None], top_idx] # (b, 20)
                    # from large to small prob
                    topn_idx = np.argsort(topn_prob)[:, ::-1] # (b, 20)
                    batch_pred_item_list = top_idx[np.arange(len(top_idx))[:, None], topn_idx] # (b, 20)
                    batch_uid = uid.cpu().data.numpy()
                    if i == 0:
                        pred_item_list = batch_pred_item_list
                        ground_truth_list = ground_truth.cpu().data.numpy()

                        target_bs_list = target_bs.cpu().data.numpy() 
                        uid_list = batch_uid
                    else:
                        pred_item_list = np.append(pred_item_list, batch_pred_item_list, axis=0)
                        ground_truth_list = np.append(ground_truth_list, ground_truth.cpu().data.numpy(), axis=0)

                        target_bs_list = np.append(target_bs_list, target_bs.cpu().data.numpy(), axis=0)
                        uid_list = np.append(uid_list, batch_uid, axis=0)

            return self.calculate_eval_metrics(mode, self.print_out_epoch,
                                               pred_item_list, ground_truth_list)
   
    def iteration(self,dataloader,epoch,early_stopping):
        # ------ model training -----#   
        bce_total_loss = 0.0
        joint_total_loss = 0.0
        cl_total_loss = 0.0
        bce_behavior_total_loss = 0.0
        iter_data = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (id_tensors,aug_tensors) in iter_data:
            '''
            user_id: (b, 1)
            input_item: (b, L)
            input_bs: (b, L, bt)
            target_item: (b, L)
            target_bs: (b, L, bt)
            target_neg: (b, L)
            ground_truth: (b, 1)
            aug_tensors: (b,2,L,bt)
            '''

            id_tensors = tuple(t.to(self.device) for t in id_tensors)
            aug_tensors = [tuple(t.to(self.device) for t in tup) for tup in aug_tensors]
            (uid, input_item, input_bs,
                target_item, target_bs, target_neg, ground_truth) = id_tensors
            sequence_output = self.model(uid, input_item, input_bs, target_bs)
            rec_loss = self.BCELoss(sequence_output, target_item, target_neg,target_bs)

            # ---------- contrastive learning task -------------#
            cl_losses = []
            if self.config['contrast_type'] == "InstanceCL":
                cl_loss = self._instance_cl_one_pair_contrastive_learning(
                    (id_tensors,aug_tensors))
                cl_losses.append(self.config['cf_weight'] * cl_loss)
            joint_loss = self.config['rec_weight'] * rec_loss
            for cl_loss in cl_losses:
                joint_loss += cl_loss
            self.optimizer.zero_grad()
            joint_loss.backward()
            self.optimizer.step()


            bce_total_loss += rec_loss.item()
            cl_total_loss += sum(cl_losses)
            joint_total_loss += joint_loss.item()

        bce_avg_loss = bce_total_loss / len(iter_data)
        cl_avg_loss = cl_total_loss / len(iter_data)
        joint_avg_loss = joint_total_loss / len(iter_data)

        self.loss_history['epochs'].append(epoch + 1)
        self.loss_history['bce_loss'].append(bce_avg_loss)
        self.loss_history['joint_loss'].append(joint_avg_loss)
        self.loss_history['cl_loss'].append(cl_avg_loss)
        self.scheduler.step(joint_avg_loss)
        
        loss_info = {
            "Epoch": epoch + 1,
            "BCE Loss": "{:.6f}".format(bce_avg_loss),
            "CL Loss": "{:.6f}".format(cl_avg_loss),
            "Joint Loss": "{:.6f}".format(joint_avg_loss)
        }

        if (epoch+1) % self.log_freq == 0:
            print(set_color(str(loss_info), "yellow"))

        if (epoch+1) % self.eval_freq == 0:
            self.print_out_epoch = epoch + 1
            scores, _ = self.valid()
            early_stopping(scores, epoch + 1, self.model)

            if early_stopping.early_stop:
                print("Early Stopping")

                best_scores_info = {
                    "HR@5": "{:.4f}".format(early_stopping.best_scores[0]),
                    "NDCG@5": "{:.4f}".format(early_stopping.best_scores[1]),
                    "HR@10": "{:.4f}".format(early_stopping.best_scores[2]),
                    "NDCG@10": "{:.4f}".format(early_stopping.best_scores[3]),
                    "HR@20": "{:.4f}".format(early_stopping.best_scores[4]),
                    "NDCG@20": "{:.4f}".format(early_stopping.best_scores[5]),
                }


                print(set_color(f'\nBest Valid (' +
                                str(early_stopping.best_valid_epoch) +
                                ') Scores: ' +
                                str(best_scores_info) + '\n', 'cyan'))
                return True
            self.model.train()
        return False