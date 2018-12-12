
import tensorflow as tf

from bert.tokenization import FullTokenizer
from tqdm import tqdm
import numpy as np

from .model_fn import BertMultiTask
from .input_fn import predict_input_fn
from .estimator import Estimator
from .utils import get_or_make_label_encoder
from .params import Params


class PredictModel():
    def __init__(self, params, model_dir=None, gpu=1):

        self.model_dir = model_dir
        self.params = params
        self.gpu = gpu
        self.tokenizer = FullTokenizer(self.params.vocab_file)

    @property
    def label_encoder(self):
        return get_or_make_label_encoder(self.problem, 'predict')

    def init_estimator(self, problem):
        self.params.assign_problem(problem, gpu=int(self.gpu))

        # change max length
        self.params.max_seq_len = 250

        model = BertMultiTask(params=self.params)
        model_fn = model.get_model_fn(warm_start=False)

        dist_trategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus=int(self.gpu),
            cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
                'nccl', num_packs=int(self.gpu)))

        run_config = tf.estimator.RunConfig(
            train_distribute=dist_trategy,
            eval_distribute=dist_trategy,
            log_step_count_steps=self.params.log_every_n_steps)

        model_dir = self.model_dir if self.model_dir is not None else self.params.ckpt_dir

        self.estimator = Estimator(
            model_fn,
            model_dir=model_dir,
            params=self.params,
            config=run_config)

    def remove_special_tokens(self, l1, l2):
        ind_list = []
        for ind, char in enumerate(l1):
            if char in ['[PAD]', '[CLS]', '[SEP]']:
                ind_list.append(ind)
        return [e for ie, e in enumerate(l1) if ie not in ind_list], [e for ie, e in enumerate(l2) if ie not in ind_list]

    def predict(self, input_file_or_list):
        def input_fn(): return predict_input_fn(
            input_file_or_list, self.params, mode='predict')

        pred = self.estimator.predict(
            input_fn=input_fn)
        return pred


class ChineseNER(PredictModel):

    def __init__(self, params, model_dir=None, gpu=1):
        super().__init__(params, model_dir, gpu)
        self.problem = 'NER'
        self.init_estimator(self.problem)

    def merge_entity(self, tokens, labels):
        merged_tokens = []
        merged_labels = []
        for token, label in zip(tokens, labels):
            if label == 'O':
                merged_tokens.append(token)
                merged_labels.append(label)
            elif label[0] == 'B':
                merged_tokens.append(token)
                merged_labels.append(label[2:])
            else:
                merged_tokens[-1] += token
                # merged_labels[-1] += label
        return merged_tokens, merged_labels

    def ner(self, input_file_or_list, extract_ent=True):

        pred = self.predict(input_file_or_list)

        result_list = []

        for d, p in enumerate(pred):
            input_ids = p['input_ids']
            ner_pred = p[self.problem]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            labels = self.label_encoder.inverse_transform(ner_pred)

            tokens, labels = self.remove_special_tokens(tokens, labels)
            tokens, labels = self.merge_entity(tokens, labels)
            if extract_ent:

                result_list.append([(ent, ent_type) for ent, ent_type in zip(
                    tokens, labels) if ent_type != 'O'])

            else:
                result_list.append(
                    list(zip(tokens, labels)))
        return result_list


class ChineseWordSegment(PredictModel):
    def __init__(self, params, model_dir=None, gpu=1):
        super().__init__(params, model_dir, gpu)
        self.problem = 'CWS'
        self.init_estimator(self.problem)

    def cws(self, input_file_or_list):
        pred = self.predict(input_file_or_list)
        result_list = []

        for d, p in enumerate(pred):
            input_ids = p['input_ids']
            ner_pred = p[self.problem]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            labels = self.label_encoder.inverse_transform(ner_pred)
            tokens, labels = self.remove_special_tokens(tokens, labels)
            output_str = ''
            for char, char_label in zip(tokens, labels):
                if char_label in ['s', 'e']:
                    output_str += char + ' '
                else:
                    output_str += char
            result_list.append(output_str)
        return result_list
