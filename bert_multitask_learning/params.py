import json
from json import decoder
import os
import re
import shutil
import logging


from .utils import create_path, load_transformer_tokenizer, load_transformer_config


class BaseParams():
    # pylint: disable=attribute-defined-outside-init
    def __init__(self):
        self.run_problem_list = []

        self.problem_type = {
        }

        # transformers params
        self.transformer_model_name = 'bert-base-chinese'
        self.transformer_tokenizer_name = 'bert-base-chinese'
        self.transformer_config_name = 'bert-base-chinese'
        self.transformer_model_loading = 'TFAutoModel'
        self.transformer_config_loading = 'AutoConfig'
        self.transformer_tokenizer_loading = 'AutoTokenizer'
        self.transformer_decoder_model_name = None
        self.transformer_decoder_config_name = None
        self.transformer_decoder_tokenizer_name = None
        # self.transformer_decoder_model_name = "hfl/chinese-xlnet-base"
        # self.transformer_decoder_config_name = "hfl/chinese-xlnet-base"
        # self.transformer_decoder_tokenizer_name = "hfl/chinese-xlnet-base"
        self.transformer_decoder_model_loading = 'TFAutoModel'
        self.transformer_decoder_config_loading = 'AutoConfig'
        self.transformer_decoder_tokenizer_loading = 'AutoTokenizer'

        # multimodal params
        self.modal_token_type_id = {
            'text': 0,
            'image': 2,
            'others': 3
        }
        # bert config
        self.init_checkpoint = ''

        # specify this will make key reuse values top
        # that it, weibo_ner problem will use NER's top
        self.share_top = {
        }
        for p in self.problem_type:
            if p not in self.share_top:
                self.share_top[p] = p

        self.multitask_balance_type = 'data_balanced'
        # self.multitask_balance_type = 'problem_balanced'

        # logging control
        self.log_every_n_steps = 100
        self.detail_log = True

        self.multiprocess = True
        self.num_cpus = 4
        self.per_cpu_buffer = 3000
        self.decode_vocab_file = None
        self.eval_throttle_secs = 600

        # training
        self.init_lr = 2e-5
        self.batch_size = 32
        self.train_epoch = 15
        self.freeze_step = 0
        self.prefetch = 5000
        self.dynamic_padding = True
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

        # experimental multitask approach
        self.label_transfer = False
        # train mask lm and downstream task at the same time
        self.augument_mask_lm = False
        self.augument_rate = 0.5
        # NOT implemented
        self.distillation = False
        # Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
        # ref: https://arxiv.org/abs/1705.07115
        self.uncertain_weight_loss = False
        # dep since not good
        # self.mutual_prediction = False

        # add an extra attention for each task
        #   with BERT layers as encoder output, task logits as decoder inputs
        self.grid_transformer = False

        # add an extra attention for each task
        #   with other tasks' logits as encoder output, task logits asn decoder inputs
        self.task_transformer = False

        # do a mean for gradients of BERT layers instead of sum
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

        # pretrain hparm
        self.dupe_factor = 10
        self.short_seq_prob = 0.1
        self.masked_lm_prob = 0.15
        self.max_predictions_per_seq = 20
        self.mask_lm_hidden_size = 768
        self.mask_lm_hidden_act = 'gelu'
        self.mask_lm_initializer_range = 0.02

        self.train_problem = None
        self.tmp_file_dir = 'tmp'
        self.cache_dir = 'models/transformers_cache'
        # get generator function for each problem
        self.read_data_fn = {}
        self.problem_assigned = False

    def add_problem(self, problem_name, problem_type='cls', processing_fn=None, share_top=None):
        if problem_type not in ['cls', 'seq_tag', 'seq2seq_tag', 'seq2seq_text', 'multi_cls', 'pretrain']:
            raise ValueError('Provided problem type not valid, expect {0}, got {1}'.format(
                ['cls', 'seq_tag', 'seq2seq_tag', 'seq2seq_text', 'multi_cls', 'pretrain'], problem_type))

        self.problem_type[problem_name] = problem_type
        self.read_data_fn[problem_name] = processing_fn
        if share_top is not None:
            self.share_top[problem_name] = share_top
        else:
            self.share_top[problem_name] = problem_name

    def assign_problem(self, flag_string: str, gpu=2, base_dir=None, dir_name=None, is_serve=False):
        """Assign the actual run problem to param. This function will
        do the following things:

        1. parse the flag string to form the run_problem_list
        2. create checkpoint saving path
        3. calculate total number of training data and training steps
        4. scale learning rate with the number of gpu linearly

        Arguments:
            flag_string {str} -- run problem string
            example: cws|POS|weibo_ner&weibo_cws

        Keyword Arguments:
            gpu {int} -- number of gpu use for training, this
                will affect the training steps and learning rate (default: {2})
            base_dir {str} -- base dir for ckpt, if None,
                then "models" is assigned (default: {None})
            dir_name {str} -- dir name for ckpt, if None,
                will be created automatically (default: {None})
        """
        self.assigned_details = (
            flag_string, gpu, base_dir, dir_name, is_serve)
        self.problem_assigned = True
        self.is_serve = is_serve

        self.problem_list, self.problem_chunk = self.parse_problem_string(
            flag_string)

        # create dir and get vocab, config
        self.prepare_dir(base_dir, dir_name, self.problem_list)

        self.get_data_info(self.problem_list, self.ckpt_dir)

        if not is_serve:
            self.shuffle_buffer = min([200000, self.data_num])
            for problem in self.problem_list:
                if self.problem_type[problem] == 'pretrain':
                    dup_fac = self.dupe_factor
                    break
                else:
                    dup_fac = 1
            self.train_steps = int((
                self.data_num * self.train_epoch * dup_fac) / (self.batch_size*max(1, gpu)))
            self.num_warmup_steps = int(0.1 * self.train_steps)

            # linear scale learing rate
            self.lr = self.init_lr * gpu

    def to_json(self):
        dump_dict = {}
        for att_name, att in vars(self).items():
            try:
                json.dumps(att)
                dump_dict[att_name] = att
            except TypeError:
                pass

        with open(self.params_path, 'w', encoding='utf8') as f:
            json.dump(dump_dict, f)

    def from_json(self, json_path=None):
        try:
            params_path = json_path if json_path is not None else self.params_path
        except AttributeError:
            raise AttributeError(
                'Either json_path should not be None or problem is assigned.')
        if self.problem_assigned:
            assign_details = self.assigned_details

        with open(params_path, 'r', encoding='utf8') as f:
            dump_dict = json.load(f)
        for att in dump_dict:
            setattr(self, att, dump_dict[att])
        self.bert_config = load_transformer_config(
            self.bert_config_dict, self.transformer_config_loading)
        if hasattr(self, 'bert_decoder_config_dict'):
            self.bert_decoder_config = load_transformer_config(
                self.bert_decoder_config_dict, self.transformer_decoder_config_loading
            )
        self.assign_problem(*assign_details)

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
        else:
            self.data_num_dict = {}
            self.num_classes = {}

        if not self.is_serve:
            # update data_num and train_steps
            self.data_num = 0
            for problem in problem_list:
                if problem not in self.data_num_dict:

                    self.data_num_dict[problem], self.num_classes[problem] = self.read_data_fn[problem](
                        self, 'train', get_data_num=True)
                    self.data_num += self.data_num_dict[problem]
                else:
                    self.data_num += self.data_num_dict[problem]

            data_info = {
                'data_num': self.data_num_dict,
                'num_classes': self.num_classes,
            }

            json.dump(data_info, open(json_path, 'w', encoding='utf8'))
        return json_path

    def parse_problem_string(self, flag_string):
        '''Parse problem string
        Example:
            cws|POS|weibo_ner&weibo_cws

            self.run_problem_list = [{cws:seq_tag}, {POS:seq_tag}, {weibo_ner:seq_tag, weibo_cws:seq_tag}]
            problem_list = [cws, POS, weibo_ner, weibo_cws]

        Arguments:
            flag_string {str} -- problem string

        Returns:
            list -- problem list
        '''

        self.problem_str = flag_string
        # Parse problem string
        self.run_problem_list = []
        problem_chunk = []
        for flag_chunk in flag_string.split('|'):

            if '&' not in flag_chunk:
                problem_type = {}
                problem_type[flag_chunk] = self.problem_type[flag_chunk]
                self.run_problem_list.append(problem_type)
                problem_chunk.append([flag_chunk])
            else:
                problem_type = {}
                problem_chunk.append([])
                for problem in flag_chunk.split('&'):
                    problem_type[problem] = self.problem_type[problem]
                    problem_chunk[-1].append(problem)
                self.run_problem_list.append(problem_type)
        # if (self.label_transfer or self.mutual_prediction) and self.train_problem is None:
        if self.train_problem is None:
            self.train_problem = [p for p in self.run_problem_list]

        problem_list = sorted(re.split(r'[&|]', flag_string))
        return problem_list, problem_chunk

    def prepare_dir(self, base_dir, dir_name, problem_list):
        base = base_dir if base_dir is not None else 'models'

        dir_name = dir_name if dir_name is not None else '_'.join(
            problem_list)+'_ckpt'
        self.ckpt_dir = os.path.join(base, dir_name)

        if not self.is_serve:
            create_path(self.ckpt_dir)

            # two ways to init model
            # 1. init from TF checkpoint dir. The dir has to contain bert_config.json.
            # 2. init from huggingface checkpoint.
            self.params_path = os.path.join(self.ckpt_dir, 'params.json')
            config_path = os.path.join(self.init_checkpoint,
                                       'bert_config.json')
            decoder_config_path = os.path.join(self.init_checkpoint,
                                               'bert_decoder_config.json')
            # bert config exists, init from existing config
            if os.path.exists(config_path):
                shutil.copy2(config_path, self.ckpt_dir)
                self.bert_config = load_transformer_config(
                    config_path, self.transformer_config_loading)
                if os.path.exists(decoder_config_path):
                    self.bert_decoder_config = load_transformer_config(
                        decoder_config_path, self.transformer_decoder_config_loading
                    )
                    self.bert_decoder_config_dict = self.bert_decoder_config.to_dict()
                self.init_weight_from_huggingface = False
            else:
                # load config from huggingface
                logging.warning(
                    '%s not exists. will load model from huggingface checkpoint.', config_path)
                # get or download config
                self.init_weight_from_huggingface = True
                self.bert_config = load_transformer_config(
                    self.transformer_config_name, self.transformer_config_loading)
                json.dump(self.bert_config.to_dict(), open(
                    os.path.join(self.ckpt_dir, 'bert_config.json'), 'w'))

                # if decoder is specified
                if self.transformer_decoder_model_name:
                    self.bert_decoder_config = load_transformer_config(
                        self.transformer_decoder_config_name, self.transformer_decoder_config_loading
                    )
                    self.bert_decoder_config_dict = self.bert_decoder_config.to_dict()
                    json.dump(self.bert_decoder_config.to_dict(),
                              open(decoder_config_path, 'w'))

        self.bert_config_dict = self.bert_config.to_dict()

        tokenizer = load_transformer_tokenizer(
            self.transformer_tokenizer_name, self.transformer_tokenizer_loading)
        self.vocab_size = tokenizer.vocab_size
        if self.transformer_decoder_tokenizer_name:
            decoder_tokenizer = load_transformer_tokenizer(
                self.transformer_decoder_tokenizer_name,
                self.transformer_decoder_tokenizer_loading
            )
            self.decoder_vocab_size = decoder_tokenizer.vocab_size
            self.bos_id = decoder_tokenizer.bos_token_id
            self.eos_id = decoder_tokenizer.eos_token_id

    def get_problem_type(self, problem: str):
        return self.problem_type[problem]


class CRFParams(BaseParams):
    def __init__(self):
        super(CRFParams, self).__init__()
        self.crf = True


class StaticBatchParams(BaseParams):
    def __init__(self):
        super(StaticBatchParams, self).__init__()
        self.dynamic_padding = False


class DynamicBatchSizeParams(BaseParams):
    def __init__(self):
        super(DynamicBatchSizeParams, self).__init__()
        self.bucket_batch_sizes = [128, 64, 32, 16]
