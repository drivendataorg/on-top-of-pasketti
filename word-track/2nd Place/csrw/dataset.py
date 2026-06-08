import types

import os, sys, logging
import json
import resource
from tqdm import tqdm
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from glob import glob
from functools import partial
from itertools import islice
import re
import math
import util
import types
#import albumentations as A
from copy import deepcopy
import torch
import torch.distributed as dist
from torch.utils.data.dataloader import RandomSampler, default_collate
from torch.utils.data.distributed import DistributedSampler
import psutil
from collections import namedtuple
from PIL import Image
from scipy.special import softmax

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class Sampler(torch.utils.data.Sampler):
    def __init__(self, cfg, data_type, ds):
        self.cfg = cfg
        self.data_type = data_type
        self.ds = ds
        self.inds = np.arange(len(self.ds.data))
        ind_num = defaultdict(int)
        for rec in self.ds.data:
            ind_num[rec.ind] += 1
        weights = {k:1/v for k, v in ind_num.items()}
        self.weights = [weights[rec.ind] for rec in self.ds.data]
        self.weights = np.array(self.weights)/np.sum(self.weights)

        assert abs(1-sum(self.weights))<1e-10

    def __len__(self):
        return self.cfg.n_sample_epoch

    def __iter__(self):
        for ind in self.gen_inds():
            yield ind

    def gen_inds(self):
        if self.data_type=='train':
            inds = np.random.choice(self.inds, self.__len__(), p=self.weights)
        else:
            raise NotImplementedError(self.data_type)
        return inds



class DatasetMixBase():
    def __init__(self, cfg, data_type, data, tokenizer=None, model_config=None, processor=None):
        self.cfg = cfg
        self.data_type = data_type
        self.tokenizer=tokenizer
        self.model_config = model_config
        self.processor = processor


    def __len__(self):
        num = len(self.data)
        if self.data_type=='train':
            num *= self.cfg.n_repeat
        return num

    def sample_item(self, index):
        for i in range(10):
            index2 = np.random.randint(self.__len__())
            if index2!=index:
                break
        item = self._getitem(index2)
        return item

    def __getitem__(self, index):
        item = self.getitem(index)
        return item

    def preprocess_data(self, data):
        data = data.to_records(index=False)
        logger.info('num of data %s is %s', self.data_type, len(data))
        return data

    def _getitem(self, index, rec=None):
        pass

    def getitem(self, index, rec=None):
        item = self._getitem(index, rec=rec)
        return item

    def collate(self, batch):
        new_batch = dict()
        batch = default_collate(batch)
        batch.update(new_batch)
        return batch



if __name__ == '__main__':
    import util
    args = util.parser.parse_args([])
