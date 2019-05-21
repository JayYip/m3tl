import os
import re
import json
import shutil

from .bert.modeling import BertConfig

from . import data_preprocessing
from .utils import create_path, EOS_TOKEN, get_or_make_label_encoder


class BaseParams():
    def __init__(self):
        self.run_problem_list = []

        self.problem_type = {
            'weibo_ner': 'seq_tag',
            'weibo_fake_cls': 'cls',
            'weibo_cws': 'seq_tag',
            'weibo_pretrain': 'pretrain',
            'CWS': 'seq_tag',
            'NER': 'seq_tag',
            'ctb_pos': 'seq_tag',
            'ctb_cws': 'seq_tag',
            'as_cws': 'seq_tag',
            'msr_cws': 'seq_tag',
            'pku_cws': 'seq_tag',
            'city_cws': 'seq_tag',
            'boson_ner': 'seq_tag',
            'msra_ner': 'seq_tag',
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
            'cityu_domain': 'cls',
            'emotion_analysis': 'cls',
            'ontonotes_pos': 'seq_tag'
        }

        # specify this will make key reuse values top
        # that it, weibo_ner problem will use NER's top
        self.share_top = {
            'ctb_cws': 'CWS',
            'as_cws': 'CWS',
            'msr_cws': 'CWS',
            'pku_cws': 'CWS',
            'city_cws': 'CWS',
            'ctb_pos': 'POS',
            'boson_domain': 'ner_domain',
            'Weibo_domain': 'ner_domain',
            'msra_domain': 'ner_domain',
            'as_domain': 'cws_domain',
            'msr_domain': 'cws_domain',
            'pku_domain': 'cws_domain',
            'cityu_domain': 'cws_domain'
        }
        for p in self.problem_type:
            if p not in self.share_top:
                self.share_top[p] = p

        self.multitask_balance_type = 'data_balanced'
        # self.multitask_balance_type = 'problem_balanced'

        # logging control
        self.log_every_n_steps = 100
        self.detail_log = True

        # training
        self.init_lr = 2e-5
        self.batch_size = 32
        self.train_epoch = 15
        self.freeze_step = 0
        self.prefetch = 5000
        self.dynamic_padding=True
        self.bucket_batch_sizes = [32, 32, 32, 16]
        self.bucket_boundaries = [30, 64, 128]

        # hparm
        self.dropout_keep_prob = 0.9
        self.max_seq_len = 256
        self.use_one_hot_embeddings = True
        self.label_smoothing = 0.0
        self.crf = False
        self.bert_num_hidden_layer = 12
        self.hidden_dense = False

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
        # dep since not good
        # self.mutual_prediction = False
        self.grid_transformer = False
        self.task_transformer = False
        self.mean_gradients = False

        # random replace punctuation by some prob to
        # ease the punctuation sensitive problem
        self.punc_replace_prob = 0.0
        self.punc_list = list(',.!?！。？，、')
        self.hidden_gru = False
        self.label_transfer_gru = False
        # if None, we will use the same hidden_size as inputs
        # e.g. # of labels
        self.label_transfer_gru_hidden_size = None

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

        self.train_problem = None
        # get generator function for each problem
        self.read_data_fn = {}
        for problem in self.problem_type:
            try:
                self.read_data_fn[problem] = getattr(
                    data_preprocessing, problem)
            except AttributeError:
                raise AttributeError(
                    '%s function not implemented in data_preprocessing.py' % problem)
        self.problem_assigned = False

    def add_problem(self, problem_name, problem_type='cls', processing_fn=None, share_top=None):
        if problem_type not in ['cls', 'seq_tag', 'seq2seq_tag', 'seq2seq_text']:
            raise ValueError('Provided problem type not valid, expect {0}, got {1}'.format((['cls', 'seq_tag', 'seq2seq_tag', 'seq2seq_text'], problem_type)))

        self.problem_type[problem_name] = problem_type
        self.read_data_fn[problem_name] = processing_fn
        if share_top is not None:
            self.share_top[problem_name] = share_top
        else:
            self.share_top[problem_name] = problem_name

    def assign_problem(self, flag_string: str, gpu=2, base_dir=None, dir_name=None):
        """Assign the actual run problem to param. This function will
        do the following things:

        1. parse the flag string to form the run_problem_list
        2. create checkpoint saving path
        3. calculate total number of training data and training steps
        4. scale learning rate with the number of gpu linearly

        Arguments:
            flag_string {str} -- run problem string
            example: CWS|POS|weibo_ner&weibo_cws

        Keyword Arguments:
            gpu {int} -- number of gpu use for training, this
                will affect the training steps and learning rate (default: {2})
            base_dir {str} -- base dir for ckpt, if None,
                then "models" is assigned (default: {None})
            dir_name {str} -- dir name for ckpt, if None,
                will be created automatically (default: {None})
        """
        self.problem_assigned = True

        self.problem_list = self.parse_problem_string(flag_string)
        # create dir and get vocab, config
        self.prepare_dir(base_dir, dir_name, self.problem_list)

        self.get_data_info(self.problem_list, self.ckpt_dir)

        self.shuffle_buffer = min([200000, self.data_num])
        for problem in self.problem_list:
            if self.problem_type[problem] == 'pretrain':
                dup_fac = self.dupe_factor
                break
            else:
                dup_fac = 1
        self.train_steps = int((
            self.data_num * self.train_epoch * dup_fac) / (self.batch_size*gpu))
        self.num_warmup_steps = int(0.1 * self.train_steps)

        # linear scale learing rate
        self.lr = self.init_lr * gpu


    @property
    def features_to_dump(self):
        # training
        return [
                'multitask_balance_type',
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
                'crf',
                'bert_num_hidden_layer',

                'decoder_num_hidden_layers',
                'beam_size',
                'init_decoder_from_encoder',
                'beam_search_alpha',
                'decode_max_seq_len',
                'label_transfer',
                'augument_mask_lm',
                'augument_rate',
                'distillation',

                # pretrain hparm
                'dupe_factor',
                'short_seq_prob',
                'masked_lm_prob',
                'max_predictions_per_seq',
                'mask_lm_hidden_size',
                'mask_lm_hidden_act',
                'mask_lm_initializer_range',
                'punc_replace_prob',
                'hidden_gru',
                'label_transfer_gru',
                'label_transfer_gru_hidden_size',
                # 'mutual_prediction',
                'grid_transformer',
                'hidden_dense',
                'task_transformer',
                'train_problem',
                'mean_gradients',
                'dynamic_padding']

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
        self.bert_config.num_hidden_layers = dump_dict['bert_num_hidden_layer']

    def get_data_info(self, problem_list, base):
        '''Get number of data, number of classes of data and eos_id of data.

        Arguments:
            problem_list {list} -- problem list
            base {str} -- path to store data_info.json
        '''

        json_path = os.path.join(base, 'data_info.json')
        if os.path.exists(json_path):
            data_info = json.load(open(json_path, 'r', encoding='utf8'))
            self.data_num_dict = data_info['data_num']
            self.num_classes = data_info['num_classes']
            self.eos_id = data_info['eos_id']
        else:
            self.data_num_dict = {}
            self.num_classes = {}
            self.eos_id = {}

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

        data_info = {
            'data_num': self.data_num_dict,
            'num_classes': self.num_classes,
            'eos_id': self.eos_id
        }

        json.dump(data_info, open(json_path, 'w', encoding='utf8'))
        return json_path

    def parse_problem_string(self, flag_string):
        '''Parse problem string
        Example:
            CWS|POS|weibo_ner&weibo_cws

            self.run_problem_list = [{CWS:seq_tag}, {POS:seq_tag}, {weibo_ner:seq_tag, weibo_cws:seq_tag}]
            problem_list = [CWS, POS, weibo_ner, weibo_cws]

        Arguments:
            flag_string {str} -- problem string

        Returns:
            list -- problem list
        '''

        self.problem_str = flag_string
        # Parse problem string
        self.run_problem_list = []
        for flag_chunk in flag_string.split('|'):

            if '&' not in flag_chunk:
                problem_type = {}
                problem_type[flag_chunk] = self.problem_type[flag_chunk]
                self.run_problem_list.append(problem_type)
            else:
                problem_type = {}
                for problem in flag_chunk.split('&'):
                    problem_type[problem] = self.problem_type[problem]
                self.run_problem_list.append(problem_type)
        # if (self.label_transfer or self.mutual_prediction) and self.train_problem is None:
        if self.train_problem is None:
            self.train_problem = [p for p in self.run_problem_list]

        problem_list = sorted(re.split(r'[&|]', flag_string))
        return problem_list

    def prepare_dir(self, base_dir, dir_name, problem_list):
        base = base_dir if base_dir is not None else 'models'

        dir_name = dir_name if dir_name is not None else '_'.join(
            problem_list)+'_ckpt'
        self.ckpt_dir = os.path.join(base, dir_name)
        create_path(self.ckpt_dir)
        self.params_path = os.path.join(self.ckpt_dir, 'params.json')
        try:
            shutil.copy2(os.path.join(self.init_checkpoint,
                         'vocab.txt'), self.ckpt_dir)
            shutil.copy2(os.path.join(self.init_checkpoint,
                                    'bert_config.json'), self.ckpt_dir)
        except FileNotFoundError:
            pass
        self.vocab_file = os.path.join(self.ckpt_dir, 'vocab.txt')
        self.bert_config = BertConfig.from_json_file(
            os.path.join(self.ckpt_dir, 'bert_config.json'))
        self.bert_config.num_hidden_layers = self.bert_num_hidden_layer
        self.bert_config_dict = self.bert_config.__dict__
        with open(self.vocab_file, 'r', encoding='utf8') as vf:
            self.vocab_size = len(vf.readlines())

class CRFParams(BaseParams):
    def __init__(self):
        super(CRFParams, self).__init__()
        self.crf = True

class StaticBatchParams(BaseParams):
    def __init__(self):
        super(StaticBatchParams, self).__init__()
        self.dynamic_padding=False

class DynamicBatchSizeParams(BaseParams):
    def __init__(self):
        super(DynamicBatchSizeParams, self).__init__()
        self.bucket_batch_sizes = [128, 64, 32, 16]