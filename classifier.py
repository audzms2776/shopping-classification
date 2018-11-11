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
import tensorflow as tf

# import fire
import h5py
import numpy as np

from misc import get_logger, get_json
from network import TextModel

opt = get_json('./config.json')
cate1 = json.load(open('./cate1.json', encoding='utf-8'))
DEV_DATA_LIST = ['./dev.chunk.01']
DATA_ROOT = './data/train'
OUT_DIR = './model/train'


def get_sample_generator(ds, batch_size):
    left, limit = 0, ds['uni'].shape[0]

    while True:
        right = min(left + batch_size, limit)
        X = [ds[t][left:right, :] for t in ['uni', 'w_uni']]
        Y = ds['cate'][left:right]

        yield X, Y

        left = right
        if right == limit:
            left = 0


class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.num_classes = 0


    def predict(self):
        meta_path = os.path.join(DATA_ROOT, 'meta')
        meta = cPickle.loads(open(meta_path, 'rb').read())

        model_fname = os.path.join(model_root, 'model.h5')
        self.logger.info('# of classes(train): %s' % len(meta['y_vocab']))
        model = tf.keras.models.load_model(model_fname)

        test_path = os.path.join(test_root, 'data.h5py')
        test_data = h5py.File(test_path, 'r')
        test = test_data[test_div]

        test_gen = get_sample_generator(test, opt.batch_size)
        total_test_samples = test['uni'].shape[0]
        steps = int(np.ceil(total_test_samples / float(opt.batch_size)))

        pred_y = model.predict_generator(test_gen,
                                         steps=steps,
                                         workers=opt.num_predict_workers,
                                         verbose=1)

        self.write_prediction_result(test, pred_y, meta, out_path, readable=readable)


    def train(self):
        data_path = os.path.join(DATA_ROOT, 'data.h5py')
        meta_path = os.path.join(DATA_ROOT, 'meta')

        data = h5py.File(data_path, 'r')
        meta = cPickle.loads(open(meta_path, 'rb').read())

        self.logger.info('# of classes: %s' % len(meta['y_vocab']))
        self.num_classes = len(meta['y_vocab'])

        train = data['train']
        dev = data['dev']

        self.logger.info('# of train samples: %s' % train['cate'].shape[0])
        self.logger.info('# of dev samples: %s' % dev['cate'].shape[0])

        model = TextModel(self.num_classes).model

        total_train_samples = train['uni'].shape[0]
        train_gen = get_sample_generator(train, batch_size=opt['batch_size'])
        self.steps_per_epoch = int(np.ceil(total_train_samples / float(opt['batch_size'])))

        total_dev_samples = dev['uni'].shape[0]
        dev_gen = get_sample_generator(dev, batch_size=opt['batch_size'])
        self.validation_steps = int(np.ceil(total_dev_samples / float(opt['batch_size'])))

        model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=opt['num_epochs'],
                            validation_data=dev_gen,
                            validation_steps=self.validation_steps,
                            shuffle=True,
                            callbacks=[tf.keras.callbacks.ModelCheckpoint('./model.h5', verbose=1)])


if __name__ == '__main__':
    clsf = Classifier()
    
    fire.Fire({'train': clsf.train, 'predict': clsf.predict})
