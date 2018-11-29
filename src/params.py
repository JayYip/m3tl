import os

from bert.modeling import BertConfig

from . import data_preprocessing


class Params():
    def __init__(self):

        self.problem_type = {'WeiboNER': 'seq_tag',
                             'WeiboFakeCLS': 'cls',
                             'WeiboSegment': 'seq_tag'}
        # self.problem = 'cls'

        self.num_classes = {
            # num of classes of problems
            # including padding if padding is needed
            # 'seq_tag': 10
            'WeiboNER': 18,
            'WeiboFakeCLS': 2,
            'WeiboSegment': 4
        }

        # get generator function for each problem
        self.read_data_fn = {}
        for problem in self.problem_type:
            try:
                self.read_data_fn[problem] = getattr(
                    data_preprocessing, problem)
            except AttributeError:
                raise AttributeError(
                    '%s function not implemented in data_preprocessing.py' % problem)

        problem_list = sorted(self.problem_type.keys())
        self.ckpt_dir = os.path.join('tmp', '_'.join(problem_list)+'_ckpt')

        # data setting
        # self.file_pattern = 'data/weiboNER*'
        self.pretrain_ckpt = 'chinese_L-12_H-768_A-12'
        self.vocab_file = os.path.join(self.pretrain_ckpt, 'vocab.txt')

        # training
        self.init_checkpoint = self.pretrain_ckpt
        self.freeze_body = False
        self.lr = 2e-5
        self.batch_size = 32
        self.train_epoch = 30
        self.freeze_step = 50

        # hparm
        self.dropout_keep_prob = 0.9
        self.max_seq_len = 128
        self.use_one_hot_embeddings = True

        # bert config
        self.bert_config = BertConfig.from_json_file(
            os.path.join(self.pretrain_ckpt, 'bert_config.json'))

    def assign_problem(self, flag_string):
        if '&' not in flag_string:
            problem_type = {}
            problem_type[flag_string] = self.problem_type[flag_string]
            self.problem_type = problem_type
        else:
            problem_type = {}
            for problem in flag_string.split('&'):
                problem_type[problem] = self.problem_type[problem]
            self.problem_type = problem_type

        problem_list = sorted(self.problem_type.keys())
        self.ckpt_dir = os.path.join('tmp', '_'.join(problem_list)+'_ckpt')

        # update data_num and train_steps
        problem = list(self.problem_type.keys())[0]
        self.data_num = len(
            list(self.read_data_fn[problem](self, 'train')))
        self.train_steps = int((
            self.data_num * self.train_epoch) / self.batch_size)
        self.num_warmup_steps = int(0.1 * self.train_steps)
