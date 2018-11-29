import pickle
import os
import unicodedata

import subprocess
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf

from bert.tokenization import (FullTokenizer, _is_control, _is_whitespace,
                               printable_text)


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
        if zero_class is not None:
            self.encode_dict[zero_class] = 0
            self.decode_dict[0] = zero_class
            if zero_class in label_set:
                label_set.remove(zero_class)
        for l_ind, l in enumerate(label_set):

            if zero_class is not None:
                new_ind = l_ind + 1
            else:
                new_ind = l_ind
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


def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_or_make_label_encoder(problem, mode, label_list=None, zero_class='O'):
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

    problem_path = os.path.join('tmp', problem)
    create_path(problem_path)
    le_path = os.path.join(problem_path, 'lable_encoder.pkl')

    if mode == 'train' and not os.path.exists(le_path):
        label_encoder = LabelEncoder()

        label_encoder.fit(label_list, zero_class=zero_class)
        pad_ind = len(label_encoder.encode_dict)
        label_encoder.encode_dict['[PAD]'] = pad_ind
        label_encoder.decode_dict[pad_ind] = '[PAD]'
        pickle.dump(label_encoder, open(le_path, 'wb'))

    else:
        with open(le_path, 'rb') as f:
            label_encoder = pickle.load(f)

    return label_encoder


def get_dirty_text_ind(text):
    """Performs invalid character removal and whitespace cleanup on text."""

    text = [unicodedata.normalize("NFD", t) for t in text]
    output = []
    for char_ind, char in enumerate(text):
        if len(char) > 1:
            output.append(char_ind)
            continue
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            output.append(char_ind)

    return output


def create_single_problem_generator(problem,
                                    inputs_list,
                                    target_list,
                                    label_encoder,
                                    params,
                                    tokenizer,
                                    epoch):
    """Function to create iterator for single problem

    This function will:
        1. Do some text cleaning using original bert tokenizer, if
            problem type is sequential tagging, corresponding labels
            will be removed.

            Example:
                Before: inputs: ['a', '&', 'c'] target: [0, 0, 1]
                After: inputs: ['a', 'c'] target: [0, 1]
        2. Add [CLS], [SEP] tokens
        3. Padding
        4. yield result dict

    Arguments:
        problem {str} -- problem name
        inputs_list {list } -- inputs list
        target_list {list} -- target list, should have the same length as inputs list
        label_encoder {LabelEncoder} -- label encoder
        params {Params} -- params
        tokenizer {tokenizer} -- Bert Tokenizer
        epoch {int} -- Deprecate
    """

    label_padding = label_encoder.encode_dict['[PAD]']
    problem_type = params.problem_type[problem]
    is_seq = problem_type in ['seq_tag']

    for ex_index, example in enumerate(zip(inputs_list, target_list)):
        raw_inputs, raw_target = example

        inputs = '\t'.join(raw_inputs)

        # tokenizer will exclude texts not chinese or eng
        # need to do the same to inputs
        tokens_a = tokenizer.tokenize(inputs)
        dirty_ind = get_dirty_text_ind(raw_inputs)
        if is_seq:
            target = [t for t_i, t in enumerate(
                raw_target) if t_i not in dirty_ind]
        else:
            target = raw_target

        if not tokens_a:
            continue

        if is_seq:
            if len(target) != len(tokens_a):
                tf.logging.warning('Data %d broken' % ex_index)
                continue

        tokens_b = None

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > params.max_seq_len - 2:
            tokens_a = tokens_a[0:(params.max_seq_len - 2)]
            if is_seq:
                target = target[0:(params.max_seq_len - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        if is_seq:
            label_id = label_encoder.transform(target).tolist()
            label_id = [label_padding] + label_id
            label_id.append(label_padding)
        else:
            label_id = label_encoder.transform([target]).tolist()[0]
        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < params.max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            if is_seq:
                label_id.append(label_padding)

        # convert to int32
        if is_seq:
            label_id = [np.int32(i) for i in label_id]
        else:
            label_id = np.int32(label_id)

        assert len(input_ids) == params.max_seq_len
        assert len(input_mask) == params.max_seq_len
        assert len(segment_ids) == params.max_seq_len

        if ex_index < 5 and epoch == 0:
            tf.logging.debug("*** Example ***")
            tf.logging.debug("tokens: %s" % " ".join(
                [printable_text(x) for x in tokens]))
            tf.logging.debug("input_ids: %s" %
                             " ".join([str(x) for x in input_ids]))
            tf.logging.debug("input_mask: %s" %
                             " ".join([str(x) for x in input_mask]))
            tf.logging.debug("segment_ids: %s" %
                             " ".join([str(x) for x in segment_ids]))
            if is_seq:
                tf.logging.debug("%s_label_ids: %s" %
                                 (problem, " ".join([str(x) for x in label_id])))
            else:
                tf.logging.debug("%s_label_ids: %s" %
                                 (problem, str(label_id)))

        yield {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            '%s_label_ids' % problem: label_id
        }


def create_generator(params, mode, epoch):
    """Function to create iterator for multiple problem

    Arguments:
        params {Params} -- params
        mode {mode} -- mode
        epoch {int} -- epochs to run
    """

    for _ in range(epoch):
        generator_list = []
        for problem, problem_type in params.problem_type.items():
            generator_list.append(
                params.read_data_fn[problem](params, mode))

        g = zip(*generator_list)
        for instance in g:
            base_dict = {}
            base_input = {}
            for problem_dict in instance:
                base_dict.update(problem_dict)
                if not base_input:
                    base_input['input_ids'] = problem_dict['input_ids']
                else:
                    assert base_input['input_ids'] == problem_dict[
                        'input_ids'], 'Inputs id of two chained problem not aligned. Please double check!'
            yield base_dict
