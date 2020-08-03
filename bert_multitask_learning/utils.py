import pickle
import os
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np


from .bert_preprocessing.tokenization import (_is_control, FullTokenizer)
from .special_tokens import BOS_TOKEN, EOS_TOKEN


class LabelEncoder(BaseEstimator, TransformerMixin):

    def fit(self, y, zero_class=None):
        """Fit label encoder
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        self : returns an instance of self.
        """
        self.encode_dict = {}
        self.decode_dict = {}
        label_set = set(y)
        if zero_class is None:
            zero_class = '[PAD]'
        else:
            label_set.update(['[PAD]'])

        self.encode_dict[zero_class] = 0
        self.decode_dict[0] = zero_class
        if zero_class in label_set:
            label_set.remove(zero_class)

        label_set = sorted(list(label_set))

        for l_ind, l in enumerate(label_set):

            new_ind = l_ind + 1

            self.encode_dict[l] = new_ind
            self.decode_dict[new_ind] = l

        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels
        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.
        Returns
        -------
        y : array-like of shape [n_samples]
        """
        self.fit(y)
        y = self.transform(y)
        return y

    def transform(self, y):
        """Transform labels to normalized encoding.
        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.
        Returns
        -------
        y : array-like of shape [n_samples]
        """
        encode_y = []
        for l in y:
            encode_y.append(self.encode_dict[l])

        return np.array(encode_y)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.
        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        decode_y = []
        for l in y:
            decode_y.append(self.decode_dict[l])

        return np.array(decode_y)

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.decode_dict, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.decode_dict = pickle.load(f)

        self.encode_dict = {v: k for k, v in self.decode_dict.items()}


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_or_make_label_encoder(params, problem, mode, label_list=None, zero_class=None):
    """Simple function to create or load existing label encoder
    If mode is train, alway create new label_encder

    Arguments:
        problem {str} -- problem name
        mode {mode} -- mode

    Keyword Arguments:
        label_list {list} -- label list to fit the encoder (default: {None})
        zero_class {str} -- what to assign as 0 (default: {'O'})

    Returns:
        LabelEncoder -- label encoder
    """
    if label_list is None:
        return None
    problem_path = params.ckpt_dir
    create_path(problem_path)
    le_path = os.path.join(problem_path, '%s_label_encoder.pkl' % problem)
    is_seq2seq_text = params.problem_type[problem] == 'seq2seq_text'
    is_multi_cls = params.problem_type[problem] == 'multi_cls'
    is_seq2seq_tag = params.problem_type[problem] == 'seq2seq_tag'

    if mode == 'train' and not os.path.exists(le_path):

        if is_seq2seq_text:
            vocab_file = params.decode_vocab_file if params.decode_vocab_file is not None else params.vocab_file
            label_encoder = FullTokenizer(vocab_file)
            pickle.dump(label_encoder, open(le_path, 'wb'))

        elif is_multi_cls:
            label_encoder = MultiLabelBinarizer()
            label_encoder.fit(label_list)
            pickle.dump(label_encoder, open(le_path, 'wb'))

        else:
            print(problem)
            if isinstance(label_list[0], list):
                label_list = [
                    item for sublist in label_list for item in sublist]
                if is_seq2seq_tag:
                    label_list.extend([BOS_TOKEN, EOS_TOKEN])
            label_encoder = LabelEncoder()

            label_encoder.fit(label_list, zero_class=zero_class)
            label_encoder.dump(le_path)

    else:

        if is_seq2seq_text or is_multi_cls:
            label_encoder = pickle.load(open(le_path, 'rb'))
        else:
            label_encoder = LabelEncoder()
            label_encoder.load(le_path)

    if not is_seq2seq_text:
        if is_multi_cls:
            params.num_classes[problem] = label_encoder.classes_.shape[0]
        else:
            params.num_classes[problem] = len(label_encoder.encode_dict)
            if EOS_TOKEN in label_encoder.encode_dict:
                params.eos_id[problem] = int(
                    label_encoder.transform([EOS_TOKEN])[0])
    else:
        params.num_classes[problem] = len(label_encoder.vocab)
        params.eos_id[problem] = label_encoder.convert_tokens_to_ids(
            [EOS_TOKEN])

    return label_encoder


def cluster_alphnum(text: str) -> list:
    """Simple funtions to aggregate eng and number

    Arguments:
        text {str} -- input text

    Returns:
        list -- list of string with chinese char or eng word as element
    """
    return_list = []
    last_is_alphnum = False

    for char in text:
        is_alphnum = bool(re.match('^[a-zA-Z0-9\[]+$', char))
        is_right_brack = char == ']'
        if is_alphnum:
            if last_is_alphnum:
                return_list[-1] += char
            else:
                return_list.append(char)
                last_is_alphnum = True
        elif is_right_brack:
            if return_list:
                return_list[-1] += char
            else:
                return_list.append(char)
            last_is_alphnum = False
        else:
            return_list.append(char)
            last_is_alphnum = False
    return return_list


def split_label_fix(label_list: list, label_encoder: LabelEncoder) -> list:
    """A function to fix splitted label. 
    Example:
        Apple -> App# $le
        splited label_list: B-ORG -> B-ORG, B-ORG
        Fixed: B-ORG, B-ORG -> B-ORG, I-ORG

    Arguments:
        label_list {list} -- label list
        label_encoder {LabelEncoder} -- label encoder

    Returns:
        list -- fixed label list
    """
    bio_set = set(['B', 'I', 'O'])
    bmes_set = set(['B', 'M', 'E', 'S'])

    keys = list(label_encoder.encode_dict.keys())
    keys = [k.upper() for k in keys]

    def _get_position_key(k):
        if '-' in k:
            return k.split('-')[0], '-'+k.split('-')[1]
        else:
            return k, ''

    position_keys = []
    for k in keys:
        position_keys.append(_get_position_key(k)[0])
    last_label = None
    position_keys = set(position_keys)
    fixed_label_list = []
    if position_keys == bio_set:
        for l in label_list:
            if _get_position_key(l)[0].upper() == 'B' and l == last_label:
                fixed_label_list.append('I' + _get_position_key(l)[1])
            else:
                last_label = l
                fixed_label_list.append(l)
        return fixed_label_list

    elif position_keys == bmes_set:
        for l in label_list:
            if _get_position_key(l)[0].upper() == 'B' and l == last_label:
                fixed_label_list.append('M' + _get_position_key(l)[1])
            else:
                last_label = l
                fixed_label_list.append(l)
        return fixed_label_list
    else:
        return label_list


def filter_empty(input_list, target_list):
    """Filter empty inputs or targets

    Arguments:
        input_list {list} -- input list
        target_list {list} -- target list

    Returns:
        input_list, target_list -- data after filter
    """
    return_input, return_target = [], []
    for inp, tar in zip(input_list, target_list):
        if inp and tar:
            return_input.append(inp)
            return_target.append(tar)
    return return_input, return_target
