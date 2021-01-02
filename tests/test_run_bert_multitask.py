import os
import shutil
import tempfile
import bert_multitask_learning
import numpy as np
import tensorflow as tf
import runpy

from .test_base import TestBase


class BertMultitaskTest(TestBase):

    def setUp(self) -> None:
        super().setUp()
        self.tmptrimckpt = tempfile.mkdtemp()
        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        self.params.assign_problem(
            'weibo_ner&weibo_fake_cls|weibo_fake_multi_cls|weibo_masklm')
        self.train_dataset = bert_multitask_learning.train_eval_input_fn(
            params=self.params)
        self.train_inputs = next(self.train_dataset.as_numpy_iterator())

    def tearDown(self) -> None:
        super().tearDown()
        shutil.rmtree(self.tmptrimckpt)

    def test_run_bert_multitask(self):
        runpy.run_path('./tests/script_to_test_run.py')
