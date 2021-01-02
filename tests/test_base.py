import shutil
import tempfile
import unittest

import bert_multitask_learning
from bert_multitask_learning.predefined_problems import (
    get_weibo_cws_fn, get_weibo_fake_cls_fn, get_weibo_fake_multi_cls_fn,
    get_weibo_ner_fn, get_weibo_masklm)


class TestBase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tmpfiledir = tempfile.mkdtemp()
        self.tmpckptdir = tempfile.mkdtemp()
        self.prepare_params()

    def tearDown(self) -> None:
        super().tearDown()
        shutil.rmtree(self.tmpfiledir)
        shutil.rmtree(self.tmpckptdir)

    def prepare_params(self):

        self.problem_type_dict = {
            'weibo_ner': 'seq_tag',
            'weibo_cws': 'seq_tag',
            'weibo_fake_multi_cls': 'multi_cls',
            'weibo_fake_cls': 'cls',
            'weibo_masklm': 'masklm'
        }

        self.processing_fn_dict = {
            'weibo_ner': get_weibo_ner_fn(file_path='/data/bert-multitask-learning/data/ner/weiboNER*'),
            'weibo_cws': get_weibo_cws_fn(file_path='/data/bert-multitask-learning/data/ner/weiboNER*'),
            'weibo_fake_cls': get_weibo_fake_cls_fn(file_path='/data/bert-multitask-learning/data/ner/weiboNER*'),
            'weibo_fake_multi_cls': get_weibo_fake_multi_cls_fn(file_path='/data/bert-multitask-learning/data/ner/weiboNER*'),
            'weibo_masklm': get_weibo_masklm(file_path='/data/bert-multitask-learning/data/ner/weiboNER*')
        }

        self.params = bert_multitask_learning.BaseParams()
        self.params.tmp_file_dir = self.tmpfiledir
        self.params.ckpt_dir = self.tmpckptdir
        self.params.transformer_model_name = 'voidful/albert_chinese_tiny'
        self.params.transformer_config_name = 'voidful/albert_chinese_tiny'
        self.params.transformer_tokenizer_name = 'voidful/albert_chinese_tiny'
        self.params.transformer_tokenizer_loading = 'BertTokenizer'
        self.params.transformer_config_loading = 'AlbertConfig'

        self.params.add_multiple_problems(
            problem_type_dict=self.problem_type_dict, processing_fn_dict=self.processing_fn_dict)
