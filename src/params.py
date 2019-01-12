import os
import re
import json
import shutil

from .bert.modeling import BertConfig

from . import data_preprocessing
from .utils import create_path, EOS_TOKEN, get_or_make_label_encoder


class Params():
    def __init__(self):

        self.run_problem_list = []

        self.problem_type = {
            'WeiboNER': 'seq_tag',
            'WeiboFakeCLS': 'cls',
            'WeiboSegment': 'seq_tag',
            'WeiboPretrain': 'pretrain',
            'CWS': 'seq_tag',
            'NER': 'seq_tag',
            'CTBPOS': 'seq_tag',
            'CTBCWS': 'seq_tag',
            'ascws': 'seq_tag',
            'msrcws': 'seq_tag',
            'pkucws': 'seq_tag',
            'cityucws': 'seq_tag',
            'bosonner': 'seq_tag',
            'msraner': 'seq_tag',
            'POS': 'seq_tag',
            'weibo_fake_seq2seq_tag': 'seq2seq_tag',
            'weibo_fake_seq_tag': 'seq_tag',
            'ontonotes_ner': 'seq_tag',
            'ontonotes_cws': 'seq_tag',
            'ontonotes_chunk': 'seq2seq_tag',
            'boson_domain': 'cls',
            'Weibo_domain': 'cls',
            'msra_domain': 'cls',
            'as_domain': 'cls',
            'msr_domain': 'cls',
            'pku_domain': 'cls',
            'cityu_domain': 'cls'
        }
        # self.problem = 'cls'

        self.num_classes = {
            # num of classes of problems
            # including padding if padding is needed
            'WeiboNER': 10,
            'WeiboFakeCLS': 2,
            'WeiboSegment': 4,
            'next_sentence': 2,
            'CWS': 5,
            'NER': 10,
            'CTBPOS': 62,
            'CTBCWS': 5,
            'ascws': 5,
            'msrcws': 5,
            'pkucws': 5,
            'cityucws': 5,
            # 'bosonner': 10,
            # 'msraner': 10,
            'msraner': 8,
            'bosonner': 14,
            'POS': 62,
            'weibo_fake_seq2seq_tag': 4,
            'weibo_fake_seq_tag': 10,
            'ontonotes_ner': 18,
            'ontonotes_cws': 5,
            # add '[SEP]' as eos token
            'ontonotes_chunk': 192,
            'boson_domain': 3,
            'Weibo_domain': 3,
            'msra_domain': 3,
            'as_domain': 4,
            'msr_domain': 4,
            'pku_domain': 4,
            'cityu_domain': 4
        }

        self.data_num_dict = {
            'CWS': 867952,
            'NER': 60000,
            'CTBPOS': 47400,
            'CTBCWS': 47400,
            'POS': 47400,
            'ascws': 708953,
            'msrcws': 86924,
            'cityucws': 53019,
            'pkucws': 19056,
            'msraner': 46364,
            'bosonner': 30000,
            'ontonotes_ner': 39086,
            'ontonotes_cws': 39086,
            'ontonotes_chunk': 39086,
            'boson_domain': 10000,
            'msra_domain': 46364,
            'as_domain': 708953,
            'msr_domain': 86924,
            'pku_domain': 53019,
            'cityu_domain': 19056
        }

        # specify this will make key reuse values top
        # that it, WeiboNER problem will use NER's top
        self.share_top = {
            'CTBCWS': 'CWS',
            'ascws': 'CWS',
            'msrcws': 'CWS',
            'pkucws': 'CWS',
            'cityucws': 'CWS',
            'CTBPOS': 'POS',
            'boson_domain': 'ner_domain',
            'Weibo_domain': 'ner_domain',
            'msra_domain': 'ner_domain',
            'as_domain': 'cws_domain',
            'msr_domain': 'cws_domain',
            'pku_domain': 'cws_domain',
            'cityu_domain': 'cws_domain'
        }

        self.eos_id = {
        }

        self.multitask_balance_type = 'data_balanced'
        # self.multitask_balance_type = 'problem_balanced'

        # logging control
        self.log_every_n_steps = 100

        # training
        self.init_lr = 2e-5
        self.batch_size = 32
        self.train_epoch = 30
        self.freeze_step = 0
        self.prefetch = 5000
        self.shuffle_buffer = 10000
        # self.prefetch = 100
        # self.shuffle_buffer = 100
        # self.train_epoch = 100

        # hparm
        self.dropout_keep_prob = 0.9
        self.max_seq_len = 128
        self.use_one_hot_embeddings = True
        self.label_smoothing = 0.1
        self.crf = True

        # seq2seq
        self.decoder_num_hidden_layers = 3
        self.beam_size = 10
        self.init_decoder_from_encoder = False
        self.beam_search_alpha = 0.6
        self.decode_max_seq_len = 90

        # multitask training
        self.label_transfer = False
        self.augument_mask_lm = False
        self.augument_rate = 0.5
        self.distillation = False

        # bert config
        self.init_checkpoint = 'chinese_L-12_H-768_A-12'


        # pretrain hparm
        self.dupe_factor = 10
        self.short_seq_prob = 0.1
        self.masked_lm_prob = 0.15
        self.max_predictions_per_seq = 20
        self.mask_lm_hidden_size = 768
        self.mask_lm_hidden_act = 'gelu'
        self.mask_lm_initializer_range = 0.02

        # for label transfer
        # self.label_transfer = True
        # self.train_epoch = 15
        # self.init_checkpoint = 'tmp/WeiboNER_bosonner_msraner_ckpt_bk'
        # self.freeze_step = 99999
        # self.init_lr = 0.001

        # get generator function for each problem
        self.read_data_fn = {}
        for problem in self.problem_type:
            try:
                self.read_data_fn[problem] = getattr(
                    data_preprocessing, problem)
            except AttributeError:
                raise AttributeError(
                    '%s function not implemented in data_preprocessing.py' % problem)

    def assign_problem(self, flag_string: str, gpu = 2, base_dir = None, dir_name = None):
        """Assign the actual run problem to param. This function will
        do the following things:

        1. parse the flag string to form the run_problem_list
        2. create checkpoint saving path
        3. calculate total number of training data and training steps
        4. scale learning rate with the number of gpu linearly

        Arguments:
            flag_string {str} -- run problem string
            example: CWS|POS|WeiboNER&WeiboSegment

        Keyword Arguments:
            gpu {int} -- number of gpu use for training, this
                will affect the training steps and learning rate (default: {2})
            base_dir {str} -- base dir for ckpt, if None,
                then "tmp" is assigned (default: {None})
            dir_name {str} -- dir name for ckpt, if None,
                will be created automatically (default: {None})
        """

        # Parse problem string
        self.run_problem_list=[]
        for flag_chunk in flag_string.split('|'):

            if '&' not in flag_chunk:
                problem_type={}
                problem_type[flag_chunk]=self.problem_type[flag_chunk]
                self.run_problem_list.append(problem_type)
            else:
                problem_type={}
                for problem in flag_chunk.split('&'):
                    problem_type[problem]=self.problem_type[problem]
                self.run_problem_list.append(problem_type)

        problem_list=sorted(re.split(r'[&|]', flag_string))

        # create dir and get vocab, config
        base=base_dir if base_dir is not None else 'tmp'

        dir_name=dir_name if dir_name is not None else '_'.join(
            problem_list)+'_ckpt'
        self.ckpt_dir = os.path.join(base, dir_name)
        create_path(self.ckpt_dir)
        self.params_path = os.path.join(self.ckpt_dir, 'params.json')
        try:
            shutil.copy2(os.path.join(self.init_checkpoint, 'vocab.txt'), self.ckpt_dir)
            shutil.copy2(os.path.join(self.init_checkpoint,
                                    'bert_config.json'), self.ckpt_dir)
        except FileNotFoundError:
            pass
        self.vocab_file = os.path.join(self.ckpt_dir, 'vocab.txt')
        self.bert_config = BertConfig.from_json_file(
            os.path.join(self.ckpt_dir, 'bert_config.json'))
        self.bert_config_dict = self.bert_config.__dict__
        with open(self.vocab_file, 'r', encoding='utf8') as vf:
            self.vocab_size = len(vf.readlines())

        # update data_num and train_steps
        self.data_num = 0
        for problem in problem_list:
            if problem not in self.data_num_dict:
                self.data_num += len(
                    list(self.read_data_fn[problem](self, 'train')))
                self.data_num_dict[problem] = len(
                    list(self.read_data_fn[problem](self, 'train')))
            else:
                self.data_num += self.data_num_dict[problem]

        if self.problem_type[problem] == 'pretrain':
            dup_fac = self.dupe_factor
        else:
            dup_fac = 1
        self.train_steps = int((
            self.data_num * self.train_epoch * dup_fac) / (self.batch_size*gpu))
        self.num_warmup_steps = int(0.1 * self.train_steps)

        # linear scale learing rate
        self.lr = self.init_lr * gpu

        # get eos_id
        for problem in problem_list:
            try:
                le = get_or_make_label_encoder(self, problem, 'predict')
                if EOS_TOKEN in le.encode_dict:
                    self.eos_id[problem] = le.transform([EOS_TOKEN])[0]
            except FileNotFoundError:
                pass

        self.to_json()

    @property
    def features_to_dump(self):
        # training
        return [
                'init_lr',
                'batch_size',
                'train_epoch',
                'freeze_step',
                'augument_mask_lm',
                'augument_rate',
                'label_transfer',

                # hparm
                'dropout_keep_prob',
                'max_seq_len',
                'use_one_hot_embeddings',
                'label_smoothing',

                # pretrain hparm
                'dupe_factor',
                'short_seq_prob',
                'masked_lm_prob',
                'max_predictions_per_seq',
                'mask_lm_hidden_size',
                'mask_lm_hidden_act',
                'mask_lm_initializer_range',
                'multitask_balance_type',
                'run_problem_list',
                'bert_config_dict']

    def to_json(self):
        dump_dict = {}
        for att in self.features_to_dump:
            value = getattr(self, att)
            dump_dict[att] = value

        with open(self.params_path, 'w', encoding='utf8') as f:
            json.dump(dump_dict, f)

    def from_json(self, json_path=None):
        params_path = json_path if json_path is not None else self.params_path
        with open(params_path, 'r', encoding='utf8') as f:
            dump_dict = json.load(f)
        for att in dump_dict:
            setattr(self, att, dump_dict[att])
