import torch
import random
from torch.utils.data import Dataset
from data_augmentation import Random, CoOccurrence, Flip, FreqMask
import copy
import numpy as np
import pickle       
import os
def neg_sample(seq_item, itemset):
    seq_item_set = set(seq_item)
    neg_item_idx = random.randint(0, len(itemset)-1)
    neg_item = itemset[neg_item_idx]
    while neg_item in seq_item_set:
        neg_item_idx = random.randint(0, len(itemset)-1)
        neg_item = itemset[neg_item_idx]
    return neg_item

class BehaviorSetSequentialRecDataset(Dataset):
    def __init__(self, config, data, data_type="train"):
        self.config = config
        (self.train_data,
         self.valid_data,
         self.bvt_data,
         self.test_data,
         userset,
         itemset,
         self.all_behavior_type) = data
        self.dataset= config['dataset']
        self.itemset = list(itemset)
        self.userset = list(userset)
        self.data_type = data_type
        self.maxlen = config['maxlen']
        self.augment_type = config['augment_type']
        self.add_prob = config['add_prob']
        self.click_prob = config['click_prob']
        self.mask_ratio = config['mask_ratio']
        self.time_prob = config['time_prob']
        if self.data_type == "train":
            self.user_co_occur_matrix_dict = self._get_co_occur_matrix(cache_path=f'./data/{self.dataset}/co_occur_matrix.pkl')    
            self.user_behavior_freq_dict = self._get_behavior_freq_dict(cache_path=f'./data/{self.dataset}/behavior_freq_dict.pkl')

            self.augmentations = {"CoOccurrence":CoOccurrence(self.user_co_occur_matrix_dict,time_prob=self.time_prob), 
                                    "Flip":Flip(click_id=0, click_prob=self.click_prob), 
                                    "FreqMask":FreqMask(self.user_behavior_freq_dict, mask_ratio=self.mask_ratio), 
                                    "Random":Random(self.user_co_occur_matrix_dict, self.user_behavior_freq_dict,click_prob=self.click_prob, mask_ratio=self.mask_ratio, time_prob=self.time_prob)}
            self.base_transform = self.augmentations[self.augment_type]
        else:
            self.base_transform = None
        
    def _get_co_occur_matrix(self,cache_path=None):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        user_co_occur_matrix_dict = {}
        global_co_matrix = None
        for user_id, seq in self.train_data.items():

            if not seq:
                continue 

            behavior_len = len(seq[0][1])  
            co_matrix = np.zeros((behavior_len, behavior_len), dtype=int)

            for _, behavior_vec in seq:
                behavior_vec = np.array(behavior_vec)
                active_indices = np.where(behavior_vec == 1)[0]
                for i in active_indices:
                    for j in active_indices:
                        co_matrix[i][j] += 1

 
            np.fill_diagonal(co_matrix, 0)

            user_co_occur_matrix_dict[user_id] = co_matrix
            if global_co_matrix is None:
                global_co_matrix = co_matrix.copy()
            else:
                global_co_matrix += co_matrix

        if global_co_matrix is not None:
            np.fill_diagonal(global_co_matrix, 0)
        result = {
        "user": user_co_occur_matrix_dict,
        "global": global_co_matrix
                }
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)


        return result
    
    def _get_behavior_freq_dict(self,cache_path=None):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        user_behavior_freq_dict = {}
        global_behavior_freq = None
        for user_id, seq in self.train_data.items():
            if not seq:
                continue  

            behavior_dim = len(seq[0][1])
            behavior_freq = np.zeros(behavior_dim, dtype=int)

            for _, behavior_vec in seq:
                behavior_vec = np.array(behavior_vec)
                behavior_freq += behavior_vec 

            user_behavior_freq_dict[user_id] = behavior_freq
            if global_behavior_freq is None:
                global_behavior_freq = behavior_freq.copy()
            else:
                global_behavior_freq += behavior_freq
        result = {
        "user": user_behavior_freq_dict,
        "global": global_behavior_freq
         }
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        return result
        
    def _one_pair_data_augmentation(self, user_id, input_bs):
        if self.data_type !='train':
            return None
        augmented_seqs = []
        for _ in range(2):
            augmented_input_bs = self.base_transform(user_id, input_bs)
            cur_bs=augmented_input_bs[:-1]
            tar_bs=augmented_input_bs[1:]

            pad_len = self.maxlen - len(cur_bs)
            cur_bs = [[0] * self.all_behavior_type] * pad_len + cur_bs
            cur_bs = cur_bs[-self.maxlen :]
            tar_bs = [[0] * self.all_behavior_type] * pad_len + tar_bs
            tar_bs = tar_bs[-self.maxlen :]
            assert len(cur_bs) == self.maxlen
            cur_tensors = torch.tensor(cur_bs, dtype=torch.float)
            tar_tensors = torch.tensor(tar_bs, dtype=torch.float)
            augmented_seqs.append((cur_tensors,tar_tensors))
        return augmented_seqs
    

    def _pack_up_data_to_tensor(self, user_id, term_seq,lack=False): 
        seq_item = [term[0] for term in term_seq]
        seq_bs = [term[1] for term in term_seq]
        augmented_bs = self._one_pair_data_augmentation(user_id, seq_bs)
        input_item, input_bs = seq_item[:-1], seq_bs[:-1]
        target_item, target_bs = seq_item[1:], seq_bs[1:]
        if lack:
            ground_truth = [0]
        else:
            ground_truth = [seq_item[-1]]

        target_neg = []
        for _ in target_item:
            target_neg.append(neg_sample(seq_item, self.itemset))


        pad_len = self.maxlen - len(input_item)
        input_item = [0] * pad_len + input_item
        input_bs = [[0] * self.all_behavior_type] * pad_len + input_bs

        target_item = [0] * pad_len + target_item
        target_bs = [[0] * self.all_behavior_type] * pad_len + target_bs

        target_neg = [0] * pad_len + target_neg

        input_item = input_item[-self.maxlen:]
        target_item = target_item[-self.maxlen:]
        target_neg = target_neg[-self.maxlen:]
        input_bs = input_bs[-self.maxlen:]
        target_bs = target_bs[-self.maxlen:]

        assert len(input_item) == self.maxlen
        assert len(target_item) == self.maxlen
        assert len(target_neg) == self.maxlen

        one_id_tensors = (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_item, dtype=torch.long),
            torch.tensor(input_bs, dtype=torch.float),
            torch.tensor(target_item, dtype=torch.long),
            torch.tensor(target_bs, dtype=torch.float),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(ground_truth, dtype=torch.long)
        )
        return one_id_tensors,augmented_bs

    def __getitem__(self, index):
        user_id = index + 1
        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            train_term_list = self.train_data[user_id]
            one_id_tensors,augmented_input_bs = self._pack_up_data_to_tensor(user_id, train_term_list, lack=False)
            return (one_id_tensors,augmented_input_bs)

        elif self.data_type == "valid":
            valid_term_list = self.train_data[user_id] + self.valid_data[user_id]
            lack = len(self.valid_data[user_id]) == 0
            one_id_tensors,_ = self._pack_up_data_to_tensor(user_id, valid_term_list, lack=lack)
        else:
            test_term_list = self.train_data[user_id] + self.valid_data[user_id] + \
                              self.bvt_data[user_id] + self.test_data[user_id]
            lack = len(self.test_data[user_id]) == 0
            one_id_tensors,_ = self._pack_up_data_to_tensor(user_id, test_term_list, lack=lack)

        return one_id_tensors

    def __len__(self):
        return len(self.train_data)