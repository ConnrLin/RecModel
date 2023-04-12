'''
Author: Felix
Date: 2023-04-12 15:07:24
LastEditors: Felix
LastEditTime: 2023-04-12 15:56:32
Description: Please enter description
'''
from tqdm import trange
import numpy as np
import pickle
from pathlib import Path


class RandomNegativeSampler:
    def __init__(self, train, val, test, user_count, item_count, sample_size, save_folder):
        self.train = train
        self.val = val
        self.test = test
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size
        self.save_folder = save_folder

    def generate_negative_samples(self):
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            # if isinstance(self.train[user][1], tuple):
            #     seen = set(x[0] for x in self.train[user])
            #     seen.update(x[0] for x in self.val[user])
            #     seen.update(x[0] for x in self.test[user])
            # else:
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])

            samples = []
            for _ in range(self.sample_size):
                item = np.random.choice(self.item_count) + 1
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count) + 1
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples

    def get_negative_samples(self):
        savefile_path = self._get_save_path()
        if savefile_path.is_file():
            print('Negatives samples exist. Loading.')
            negative_samples = pickle.load(savefile_path.open('rb'))
            return negative_samples
        print("Negative samples don't exist. Generating.")
        negative_samples = self.generate_negative_samples()
        with savefile_path.open('wb') as f:
            pickle.dump(negative_samples, f)
        return negative_samples

    def _get_save_path(self):
        folder = Path(self.save_folder)
        filename = '{}-sample_size{}.pkl'.format('random', self.sample_size)
        return folder.joinpath(filename)
