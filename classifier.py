# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import pickle as cPickle

import fire
import h5py
import numpy as np

from misc import get_logger, get_json
from network import TextModel

opt = get_json('./config.json')
cate1 = json.load(open('./cate1.json', encoding='utf-8'))
DEV_DATA_LIST = ['./dev.chunk.01']
DATA_ROOT = './data/train'
OUT_DIR = './model/train'


class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.num_classes = 0

    def get_sample_generator(self, ds, batch_size):
        left, limit = 0, ds['uni'].shape[0]
        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right, :] for t in ['uni', 'w_uni']]
            Y = ds['cate'][left:right]
            yield X, Y
            left = right
            if right == limit:
                left = 0

    def predict(self):
        pass

    def train(self):
        data_path = os.path.join(DATA_ROOT, 'data.h5py')
        meta_path = os.path.join(DATA_ROOT, 'meta')

        data = h5py.File(data_path, 'r')
        meta = cPickle.loads(open(meta_path, 'rb').read())

        self.weight_fname = os.path.join(OUT_DIR, 'weights')
        self.model_fname = os.path.join(OUT_DIR, 'model')
        if not os.path.isdir(OUT_DIR):
            os.makedirs(OUT_DIR)

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.num_classes = len(meta['y_vocab'])

        train = data['train']
        dev = data['dev']

        self.logger.info('# of train samples: %s' % train['cate'].shape[0])
        self.logger.info('# of dev samples: %s' % dev['cate'].shape[0])

        model = TextModel(self.num_classes).model

        total_train_samples = train['uni'].shape[0]
        train_gen = self.get_sample_generator(train, batch_size=opt['batch_size'])

        # total_dev_samples = dev['uni'].shape[0]
        # dev_gen = self.get_sample_generator(dev, batch_size=opt['batch_size])


if __name__ == '__main__':
    clsf = Classifier()
    fire.Fire({'train': clsf.train,
               'predict': clsf.predict})
