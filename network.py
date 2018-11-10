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

import tensorflow as tf
from tensorflow import keras

from misc import get_logger, get_json
opt = get_json('./config.json')


def top1_acc(x, y):
    return tf.keras.metrics.top_k_categorical_accuracy(x, y, k=1)


class TextModel:
    def __init__(self, num_classes):
        logger = get_logger('textonly')
        max_len = opt['max_len']
        voca_size = opt['unigram_hash_size'] + 1

        t_input = tf.keras.Input(shape=(max_len,))
        tx = keras.layers.Embedding(voca_size, opt['embd_size'])(t_input)

        w_input = tf.keras.Input(shape=(max_len,))
        wx = tf.keras.layers.Reshape((max_len, 1))(w_input)

        x = tf.keras.layers.dot([tx, wx], axes=1)
        x = tf.keras.layers.Reshape((opt['embd_size'], ))(x)
        x = keras.layers.Dense(16, activation=tf.nn.relu)(x)
        outputs = keras.layers.Dense(num_classes, activation=tf.nn.sigmoid)(x)

        model = tf.keras.models.Model(inputs=[t_input, w_input], outputs=outputs)

        model.summary(print_fn=lambda x: logger.info(x))
        model.compile(loss='binary_crossentropy',
            optimizer=tf.train.AdamOptimizer(),
            metrics=[top1_acc])

        self.model = model
