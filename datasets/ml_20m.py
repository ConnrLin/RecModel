'''
Author: Felix
Date: 2023-04-12 13:47:14
LastEditors: Felix
LastEditTime: 2023-04-12 16:37:42
Description: DataSet class for MovieLens 20m dataset
'''
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
from tqdm import tqdm
tqdm.pandas()


class ML20MDataset:

    def __init__(self, file_dir, min_rating, min_sc, min_uc) -> None:
        self.file_dir = file_dir
        self.min_rating = min_rating
        self.min_sc = min_sc
        self.min_uc = min_uc

    def load_dataset(self):
        if not os.path.exists(os.path.join(self.file_dir, 'dataset.pickle')):
            self.preprocess()
        with open(os.path.join(self.file_dir, 'dataset.pickle'), 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    def preprocess(self):
        # load dataset
        df = pd.read_csv(os.path.join(self.file_dir, 'ratings.csv'))
        df.columns = ['uid', 'sid', 'rating', 'timestamp']

        # turn into implicit rating
        df = df[df['rating'] >= self.min_rating]

        # filtering items which have number of ratings more than min_sc
        if self.min_sc > 0:
            item_size = df.groupby('sid').size()
            satisified = item_size.index[item_size >= self.min_sc]
            df = df[df['sid'].isin(satisified)]

        # filtering users whos rating history more than min_uc
        if self.min_uc > 0:
            user_size = df.groupby('uid').size()
            satisified = user_size.index[user_size >= self.min_uc]
            df = df[df['uid'].isin(satisified)]

        # building maps for uid and sid
        umap = {u: i for i, u in enumerate(set(df['uid']))}
        smap = {s: i for i, s in enumerate(set(df['sid']))}

        # transform df
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)

        # split dataset to trainging sets, validations set and testing sets
        user_group = df.groupby('uid')
        user2items = user_group.progress_apply(
            lambda d: list(d.sort_values(by='timestamp')['sid']))
        train, val, test = {}, {}, {}
        for user in range(len(umap)):
            items = user2items[user]
            train[user], val[user], test[user] = items[:-
                                                       2], items[-2:-1], items[-1:]
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with open(os.path.join(self.file_dir, 'dataset.pickle'), 'wb') as f:
            pickle.dump(dataset, f)
