
import os
import pickle
import re

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer
import transformers


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
            label_encoder = load_transformer_tokenizer(
                params.transformer_decoder_tokenizer_name, params.transformer_decoder_tokenizer_loading)
            pickle.dump(label_encoder, open(le_path, 'wb'))

        elif is_multi_cls:
            label_encoder = MultiLabelBinarizer()
            label_encoder.fit(label_list)
            pickle.dump(label_encoder, open(le_path, 'wb'))

        else:
            if isinstance(label_list[0], list):
                label_list = [
                    item for sublist in label_list for item in sublist]
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
    else:
        try:
            params.num_classes[problem] = len(label_encoder.vocab)
        except AttributeError:
            # models like xlnet's vocab size can only be retrieved from config instead of tokenizer
            params.num_classes[problem] = params.bert_decoder_config.vocab_size

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


def infer_shape_and_type_from_dict(inp_dict: dict, fix_dim_for_high_rank_tensor=True):
    shape_dict = {}
    type_dict = {}
    for feature_name, feature in inp_dict.items():
        if type(feature) is list:
            feature = np.array(feature)
        if type(feature) is np.ndarray:
            if issubclass(feature.dtype.type, np.integer):
                type_dict[feature_name] = tf.int32
            elif issubclass(feature.dtype.type, np.float):
                type_dict[feature_name] = tf.float32

            # this seems not a good idea
            if len(feature.shape) > 1 and fix_dim_for_high_rank_tensor:
                shape_dict[feature_name] = [
                    None] + list(feature.shape[1:])
            else:
                shape_dict[feature_name] = [
                    None for _ in feature.shape]

        elif np.issubdtype(type(feature), np.float):

            type_dict[feature_name] = tf.float32
            shape_dict[feature_name] = []
        elif np.issubdtype(type(feature), np.integer):

            type_dict[feature_name] = tf.int32
            shape_dict[feature_name] = []
        else:
            if isinstance(feature, str):
                feature = feature.encode('utf8')

            type_dict[feature_name] = tf.string
            shape_dict[feature_name] = []
    return shape_dict, type_dict


def load_transformer_tokenizer(tokenizer_name: str, load_module_name=None):
    """some tokenizers cannot be loaded using AutoTokenizer. 

    this function served as a util function to catch that situation.

    Args:
        tokenizer_name (str): tokenizer name
    """
    if load_module_name:
        tok = getattr(transformers, load_module_name).from_pretrained(
            tokenizer_name)
    else:
        tok = AutoTokenizer.from_pretrained(tokenizer_name)

    return tok


def load_transformer_config(config_name_or_dict, load_module_name=None):
    """Some models need specify loading module

    Args:
        config_name (str): module name
        load_module_name (str, optional): loading module name. Defaults to None.

    Returns:
        config: config
    """
    if load_module_name:
        load_module = getattr(transformers, load_module_name)
    else:
        load_module = transformers.AutoConfig
    if isinstance(config_name_or_dict, str):
        config = load_module.from_pretrained(config_name_or_dict)
    elif isinstance(config_name_or_dict, dict):
        config = load_module.from_dict(config_name_or_dict)
    else:
        raise ValueError('config_name_or_dict should be str or dict')
    return config


def load_transformer_model(model_name_or_config, load_module_name=None):
    if load_module_name:
        load_module = getattr(transformers, load_module_name)
    else:
        load_module = transformers.TFAutoModel

    if isinstance(model_name_or_config, str):
        model = load_module.from_pretrained(model_name_or_config)
    else:
        model = load_module.from_config(model_name_or_config)
    return model


def get_transformer_main_model(model, key='embeddings'):
    """Function to extrac model name from huggingface transformer models.

    Args:
        model (Model): Huggingface transformers model
        key (str, optional): Key to identify model. Defaults to 'embeddings'.

    Returns:
        model
    """

    model_attr_name_list = model.__dict__.keys()
    for attr_name in model_attr_name_list:
        attr = getattr(model, attr_name)
        if hasattr(attr, key):
            return attr
