import numpy as np
import bert_multitask_learning
from bert_multitask_learning import (get_or_make_label_encoder)
import transformers
import tensorflow as tf

from .test_base import TestBase


class UtilsTest(TestBase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_get_or_make_label_encoder(self):
        # nested list train
        le_train = get_or_make_label_encoder(
            params=self.params, problem='weibo_ner', mode=bert_multitask_learning.TRAIN, label_list=[['a', 'b'], ['c']]
        )
        # seq_tag will add [PAD]
        self.assertEqual(len(le_train.encode_dict), 4)

        le_predict = get_or_make_label_encoder(
            params=self.params, problem='weibo_ner', mode=bert_multitask_learning.PREDICT)
        self.assertEqual(le_predict.encode_dict, le_train.encode_dict)

        # list train
        le_train = get_or_make_label_encoder(
            params=self.params, problem='weibo_fake_cls', mode=bert_multitask_learning.TRAIN, label_list=['a', 'b', 'c']
        )
        # seq_tag will add [PAD]
        self.assertEqual(len(le_train.encode_dict), 3)

        le_predict = get_or_make_label_encoder(
            params=self.params, problem='weibo_fake_cls', mode=bert_multitask_learning.PREDICT)
        self.assertEqual(le_predict.encode_dict, le_train.encode_dict)

        # text
        le_train = get_or_make_label_encoder(
            params=self.params, problem='weibo_masklm', mode=bert_multitask_learning.TRAIN)
        self.assertIsInstance(le_train, transformers.PreTrainedTokenizer)
        le_predict = get_or_make_label_encoder(
            params=self.params, problem='weibo_masklm', mode=bert_multitask_learning.PREDICT)
        self.assertIsInstance(le_predict, transformers.PreTrainedTokenizer)

    def test_infer_shape_and_type_from_dict(self):
        # dose not support nested dict
        test_dict = {
            'test1': np.random.uniform(size=(64, 32)),
            'test2': np.array([1, 2, 3], dtype='int32'),
            'test5': 5
        }
        desc_dict = bert_multitask_learning.infer_shape_and_type_from_dict(
            test_dict)
        self.assertEqual(desc_dict, ({'test1': [None, 32], 'test2': [None], 'test5': []}, {
                         'test1': tf.float32, 'test2': tf.int32, 'test5': tf.int32}))

    def test_load_transformer_tokenizer(self):
        bert_multitask_learning.load_transformer_tokenizer(
            'voidful/albert_chinese_tiny', 'BertTokenizer')

    def test_load_transformer_config(self):
        # load config with name
        config = bert_multitask_learning.load_transformer_config(
            'bert-base-chinese')
        config_dict = config.to_dict()
        # load config with dict
        config = bert_multitask_learning.load_transformer_config(
            config_dict, load_module_name='BertConfig')

    def test_load_transformer_model(self):
        # load by name(load weights)
        # this is a pt only model
        model = bert_multitask_learning.load_transformer_model(
            'voidful/albert_chinese_tiny')

        # load by config (not load weights)
        model = bert_multitask_learning.load_transformer_model(bert_multitask_learning.load_transformer_config(
            'bert-base-chinese'))

    def test_get_transformer_main_model(self):
        model = bert_multitask_learning.load_transformer_model(
            'voidful/albert_chinese_tiny')
        main_model = bert_multitask_learning.get_transformer_main_model(model)
        self.assertIsInstance(main_model, transformers.TFAlbertMainLayer)

    def test_get_embedding_table_from_model(self):
        model = bert_multitask_learning.load_transformer_model(
            'voidful/albert_chinese_tiny')
        embedding = bert_multitask_learning.get_embedding_table_from_model(
            model)
        self.assertEqual(embedding.shape, (21128, 128))
