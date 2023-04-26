'''
Author: Felix
Date: 2023-04-12 14:57:44
LastEditors: Felix
LastEditTime: 2023-04-26 09:09:02
Description: Data loader for Bert4rec
'''
import torch.nn as nn
from torch.utils.data import Dataset
from .utils import RandomNegativeSampler
import random
import torch


class BertDataset:
    """
        Factory to build Train, Test, Validate Dataset 
    """
    def __init__(self, dataset, save_folder, max_len):
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.max_len = max_len
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        self.rng = random.Random()
        self.PAD = 0
        self.MASK = self.item_count+1
        test_negative_sampler = RandomNegativeSampler(
            self.train, self.val, self.test, self.user_count, self.item_count, 100, save_folder)
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    def get_train_datasets(self):
        """ generate training dataset 

        Returns:
            torch.utils.data.Dataset: training set
        """
        train_dataset = BertTrainDataset(
            self.train, self.max_len, 0.15, self.MASK, self.item_count, self.rng)
        return train_dataset

    def get_eval_datasets(self, mode):
        """generate eval dataset 

        Args:
            mode (str): a string to specific the type of dataset, "test" or "val".

        Returns:
            torch.utils.data.Dataset: evaliating set
        """
        if mode == 'val':
            eval_datasets = BertEvalDataset(
                self.train, self.val, self.max_len, self.MASK, self.test_negative_samples)
        else:
            eval_datasets = BertEvalDataset(
                self.train, self.test, self.max_len, self.MASK, self.test_negative_samples)
        return eval_datasets

    def get_datasets(self):
        """ return training set, validating set and testing set

        Returns:
            torch.utils.data.Dataset
        """
        train_dataset = self.get_train_datasets()
        val_dataset = self.get_eval_datasets('val')
        test_dataset = self.get_eval_datasets('test')
        return train_dataset, val_dataset, test_dataset


class BertTrainDataset(Dataset):
    """ training dataset class
    """

    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]


class BertEvalDataset(Dataset):
    """ evaliating dataset class

    """
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)
