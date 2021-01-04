import os

import bert_multitask_learning
import numpy as np
import tensorflow as tf
import shutil

from .test_base import TestBase


class InputFnTest(TestBase):

    def test_train_eval_input_fn(self):
        self.params.assign_problem(
            'weibo_ner&weibo_fake_cls|weibo_fake_multi_cls|weibo_masklm')

        train_dataset = bert_multitask_learning.train_eval_input_fn(
            params=self.params, mode=bert_multitask_learning.TRAIN)
        eval_dataset = bert_multitask_learning.train_eval_input_fn(
            params=self.params, mode=bert_multitask_learning.EVAL
        )

        _ = next(train_dataset.as_numpy_iterator())
        _ = next(eval_dataset.as_numpy_iterator())

        # dynamic_padding disabled
        # have to remove existing tfrecord
        shutil.rmtree(self.tmpfiledir)
        self.params.dynamic_padding = False
        train_dataset = bert_multitask_learning.train_eval_input_fn(
            params=self.params, mode=bert_multitask_learning.TRAIN)
        eval_dataset = bert_multitask_learning.train_eval_input_fn(
            params=self.params, mode=bert_multitask_learning.EVAL
        )

        _ = next(train_dataset.as_numpy_iterator())
        _ = next(eval_dataset.as_numpy_iterator())

    def test_predict_input_fn(self):

        # single modal input
        self.params.assign_problem(
            'weibo_ner&weibo_fake_cls|weibo_fake_multi_cls|weibo_masklm')
        single_dataset = bert_multitask_learning.predict_input_fn(
            ['this is a test']*5, params=self.params)
        first_batch = next(single_dataset.as_numpy_iterator())
        self.assertEqual(first_batch['input_ids'].tolist()[0], [
                         101,  8554,  8310,   143, 10060,   102])

        # multi modal input
        mm_input = [{'text': 'this is a test',
                     'image': np.zeros(shape=(5, 10), dtype='float32')}] * 5
        mm_dataset = bert_multitask_learning.predict_input_fn(
            mm_input, params=self.params)
        first_batch = next(mm_dataset.as_numpy_iterator())
        self.assertEqual(first_batch['input_ids'].tolist()[0], [
                         101,  8554,  8310,   143, 10060,   102])
        self.assertEqual(first_batch['image_input'].tolist()[0], np.zeros(
            shape=(5, 10), dtype='float32').tolist())
