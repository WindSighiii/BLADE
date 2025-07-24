import copy
import random
import numpy as np

def ensure_change(max_retry=10):
    def decorator(augment_func):
        def wrapper(*args, **kwargs):
            sequence = args[-1]
            for _ in range(max_retry):
                augmented = augment_func(*args, **kwargs)
                if not np.array_equal(augmented, sequence):
                    return augmented
            return augmented
        return wrapper
    return decorator
mode='global'
class Random(object):
    def __init__(self, user_co_occur_matrix_dict, user_behavior_freq_dict, click_prob=0.2, mask_ratio=0.2, time_prob=0.5
    ):
        self.data_augmentation_methods = [CoOccurrence(user_co_occur_matrix_dict,time_prob=time_prob), 
                                          Flip(click_id=0, click_prob=click_prob), 
                                          FreqMask(user_behavior_freq_dict, mask_ratio=mask_ratio)]
    @ensure_change(max_retry=10)
    def __call__(self, user_id, sequence):
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        return augment_method(user_id, sequence)


    
class CoOccurrence(object):


    def __init__(self, user_co_occur_matrix_dict,time_prob=0.3):

        self.user_co_occur_matrix_dict = user_co_occur_matrix_dict[mode]
        self.time_prob = time_prob

    def __call__(self, user_id, sequence):

        copied_sequence = copy.deepcopy(sequence)
        L, B = len(sequence), len(sequence[0])
        if mode=='user':
            co_occur_matrix = self.user_co_occur_matrix_dict.get(user_id, np.ones((B, B)))
        else:
            co_occur_matrix = self.user_co_occur_matrix_dict


        n_time = max(1, int(self.time_prob * L))
        chosen_indices = random.sample(range(L), n_time)

        for t in chosen_indices:

            original_behaviors = np.where(sequence[t] == 1)[0]
            if len(original_behaviors) == 0:
                continue  


            co_probs = np.sum(co_occur_matrix[original_behaviors], axis=0)


            co_probs[original_behaviors] = 0

            if co_probs.sum() == 0:
                continue

            co_probs = co_probs / co_probs.sum()


            add_b = np.random.choice(B, p=co_probs)
            copied_sequence[t][add_b] = 1

        return copied_sequence

class Flip(object):

    def __init__(self, click_id=0, click_prob=0.2):

        self.click_id = click_id
        self.click_prob = click_prob

    def __call__(self, user_id, sequence):

        copied_sequence = copy.deepcopy(sequence)
        L = len(copied_sequence)

        n_flip = int(self.click_prob * L)
        if n_flip < 1:
            n_flip = 1  


        indices = random.sample(range(L), n_flip)

        for idx in indices:
            copied_sequence[idx][self.click_id] ^= 1  

        return copied_sequence
    


class FreqMask(object):
    def __init__(self,
                 user_behavior_freq_dict,
                 mask_ratio=0.2,
                 alpha=0.7,
                 p_max=0.3,
                 click_id=0):

        self.user_behavior_freq_dict = user_behavior_freq_dict[mode]
        self.mask_ratio = mask_ratio
        self.alpha = alpha
        self.p_max = p_max
        self.click_id = click_id

    def __call__(self, user_id, sequence):

        copied = copy.deepcopy(sequence)
        L, B = len(sequence), len(sequence[0])


        if mode =='user':
            freq = self.user_behavior_freq_dict.get(user_id,
                                                np.ones(B, dtype=float))
        else :
            freq=self.user_behavior_freq_dict

        freq_sm = np.power(freq, self.alpha)


        freq_sm[self.click_id] = min(freq_sm[self.click_id], self.p_max)


        maskable = freq_sm > 0
        if not np.any(maskable):
            return copied
        prob_vec = np.zeros_like(freq_sm)
        prob_vec[maskable] = freq_sm[maskable]
        prob_vec = prob_vec / prob_vec.sum()


        mask_num = int(self.mask_ratio * L)
        for b in range(B):

            idxs = np.random.choice(L, size=mask_num, replace=False)
            for t in idxs:
                if random.random() < prob_vec[b]:
                    copied[t][b] = 0
        return copied



