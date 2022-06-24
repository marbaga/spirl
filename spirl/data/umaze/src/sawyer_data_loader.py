import numpy as np
import itertools
import os.path as osp

from spirl.data.kitchen.src.kitchen_data_loader import D4RLSequenceSplitDataset
from spirl.utils.general_utils import AttrDict


class UmazeSequenceSplitDataset(D4RLSequenceSplitDataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        # self.data_dir = '~/.d4rl/datasets/maze_primitive_distilled.npy'
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len # 11
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size # 160
        self.device = data_conf.device # cuda
        self.n_worker = 4
        self.shuffle = shuffle # False

        path = osp.expanduser('~/.d4rl/datasets/maze_primitive_distilled.npy')
        if osp.isfile(path):
            self.dataset = np.load(path, allow_pickle=True).item()
        else:
            raise Exception(f'Dataset generation has not happened yet.')
        self.dataset['observations'] = np.concatenate([self.dataset['states'], self.dataset['goals']], -1)
        self.dataset = {k: v.astype(np.float32) for k, v in self.dataset.items()}

        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))
            start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
            if not isinstance(self.spec.filter_indices[0], list):
                self.spec.filter_indices = [self.spec.filter_indices]
            self.seqs = list(itertools.chain.from_iterable([\
                list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                               for x in self.seqs[fi[0] : fi[1]+1])) for fi in self.spec.filter_indices]))
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs
