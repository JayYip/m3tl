import pickle
import os
import unicodedata
import random
import collections
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np


from .tokenization import (_is_control, FullTokenizer)
from .special_tokens import *


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


def tokenize_text_with_seqs(tokenizer, inputs_a, target, is_seq=False):
    if isinstance(inputs_a, list):
        inputs_a_str = '\t'.join([t if t != '\t' else ' ' for t in inputs_a])
    else:
        inputs_a_str = inputs_a
    if is_seq:
        tokenized_inputs, target = tokenizer.tokenize(inputs_a_str, target)
    else:
        tokenized_inputs = tokenizer.tokenize(inputs_a_str)

    return (tokenized_inputs, target)


def _truncate_seq_pair(tokens_a, tokens_b, max_length, rng):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    if rng is None:
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    else:
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break

            trunc_tokens = tokens_a if len(
                tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if rng.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()


def truncate_seq_pair(tokens_a, tokens_b, target, max_length, rng=None, is_seq=False):
    if tokens_b is None:
        if len(tokens_a) > max_length - 2:
            tokens_a = tokens_a[0:(max_length - 2)]
            if is_seq:
                target = target[0:(max_length - 2)]

    else:
        _truncate_seq_pair(tokens_a, tokens_b, max_length-3, rng)

    return tokens_a, tokens_b, target


def add_special_tokens_with_seqs(
        tokens_a, tokens_b, target, is_seq=False):

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

    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if is_seq:
        target = ['[PAD]'] + target + ['[PAD]']

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    return (tokens, segment_ids, target)


def create_mask_and_padding(tokens, segment_ids, target, max_length, is_seq=False, dynamic_padding=False):

    input_mask = [1]*len(tokens)

    if not dynamic_padding:
        pad_list = ['[PAD]'] * (max_length - len(input_mask))

        input_mask += [0]*len(pad_list)
        segment_ids += [0]*len(pad_list)
        tokens += pad_list

        if is_seq:
            target += pad_list

    return input_mask, tokens, segment_ids, target


def punc_augument(raw_inputs, params):
    for char_ind, char in enumerate(raw_inputs):
        if char in params.punc_list:
            if random.uniform(0, 1) <= params.punc_replace_prob:
                raw_inputs[char_ind] = random.choice(params.punc_list)

    return raw_inputs

# some code block from run_pretraining.py


def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(
                            0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, None,
                                  max_num_tokens, rng)
                if len(tokens_a) < 1 or len(tokens_b) < 1:
                    current_chunk = []
                    current_length = 0
                    i += 1
                    continue
                assert len(tokens_a) >= 1, tokens_a
                assert len(tokens_b) >= 1, tokens_b

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                     tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])
TrainingInstance = collections.namedtuple("TrainingInstance",
                                          ['tokens', 'segment_ids',
                                           'masked_lm_positions',
                                           'masked_lm_labels',
                                           'is_random_next'])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(
                    0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


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
