import pickle
import os
import unicodedata
import random
import collections
from copy import copy


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


from .bert.tokenization import (_is_control,
                                printable_text, FullTokenizer)

BOS_TOKEN = '[PAD]'
EOS_TOKEN = '[SEP]'


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


def get_text_and_label(params, problem, mode, label_id=True):
    data = list(params.read_data_fn[problem](params, mode))
    data_input_ids = [d['input_ids'] for d in data]
    tokenizer = FullTokenizer(params.vocab_file, True)
    texts = [tokenizer.convert_ids_to_tokens(ids) for ids in data_input_ids]
    text_without_special_token = []
    for text in texts:
        text_without_special_token.append([])
        for t in text:
            if t not in ['[SEP]', '[CLS]', '[PAD]']:
                text_without_special_token[-1].append(t)
    return text_without_special_token, [d['%s_label_ids' % problem] for d in data]


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

    if mode == 'train' and not os.path.exists(le_path):
        label_encoder = LabelEncoder()

        label_encoder.fit(label_list, zero_class=zero_class)

        label_encoder.dump(le_path)

    else:
        label_encoder = LabelEncoder()
        label_encoder.load(le_path)

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
        inputs_a_str = '\t'.join(inputs_a)
    else:
        inputs_a_str = inputs_a

    tokenized_inputs = tokenizer.tokenize(inputs_a_str)
    dirty_ind = get_dirty_text_ind(inputs_a)

    # get white space ind
    dirty_ind += [i for i, c in enumerate(inputs_a) if not c.strip()]

    if is_seq:
        target = [element for element_i, element in enumerate(
            target) if element_i not in dirty_ind]

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


def create_mask_and_padding(tokens, segment_ids, target, max_length, is_seq=False):

    input_mask = [1]*len(tokens)
    pad_list = ['[PAD]'] * (max_length - len(input_mask))

    input_mask += [0]*len(pad_list)
    segment_ids += [0]*len(pad_list)
    tokens += pad_list
    if is_seq:
        target += pad_list

    return input_mask, tokens, segment_ids, target


def create_single_problem_generator(problem,
                                    inputs_list,
                                    target_list,
                                    label_encoder,
                                    params,
                                    tokenizer):
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

    problem_type = params.problem_type[problem]

    # whether this problem is sequential labeling
    # for sequential labeling, targets needs to align with any
    # change of inputs
    is_seq = problem_type in ['seq_tag']

    for ex_index, example in enumerate(zip(inputs_list, target_list)):
        raw_inputs, raw_target = example

        if isinstance(raw_inputs, dict):
            tokens_a, target = tokenize_text_with_seqs(
                tokenizer, raw_inputs['a'], raw_target, is_seq)
            tokens_b, _ = tokenize_text_with_seqs(
                tokenizer, raw_inputs['b'], raw_target)
        else:
            tokens_a, target = tokenize_text_with_seqs(
                tokenizer, raw_inputs, raw_target, is_seq)
            tokens_b = None

        if tokens_b is not None and is_seq:
            raise NotImplementedError(
                'Sequence Labeling with tokens b is not implemented')

        if not tokens_a:
            continue

        if is_seq:
            if len(target) != len(tokens_a):
                tf.logging.warning('Data %d broken' % ex_index)
                continue

        tokens_a, tokens_b, target = truncate_seq_pair(
            tokens_a, tokens_b, target, params.max_seq_len, is_seq=is_seq)

        tokens, segment_ids, target = add_special_tokens_with_seqs(
            tokens_a, tokens_b, target, is_seq)

        if params.augument_mask_lm:
            rng = random.Random()
            (mask_lm_tokens, masked_lm_positions,
                masked_lm_labels) = create_masked_lm_predictions(
                    tokens,
                    params.masked_lm_prob,
                    params.max_predictions_per_seq,
                    list(tokenizer.vocab.keys()), rng)
            _, mask_lm_tokens, _, _ = create_mask_and_padding(
                mask_lm_tokens, copy(segment_ids), copy(target), params.max_seq_len, is_seq)
            masked_lm_weights, masked_lm_labels, masked_lm_positions, _ = create_mask_and_padding(
                masked_lm_labels, masked_lm_positions, None, params.max_predictions_per_seq)
            mask_lm_input_ids = tokenizer.convert_tokens_to_ids(
                mask_lm_tokens)
            masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_mask, tokens, segment_ids, target = create_mask_and_padding(
            tokens, segment_ids, target, params.max_seq_len, is_seq)

        # create mask and padding for labels of seq2seq problem
        if problem_type in ['seq2seq_tag', 'seq2seq_text']:

            target, _, _ = truncate_seq_pair(
                target, None, None, params.decode_max_seq_len, is_seq=is_seq)
            # since we initialize the id to 0 in prediction, we need
            # to make sure that BOS_TOKEN is [PAD]
            target = [BOS_TOKEN] + target + [EOS_TOKEN]
            label_mask, target, _, _ = create_mask_and_padding(
                target, [0] * len(target), None, params.decode_max_seq_len)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if isinstance(target, list):
            label_id = label_encoder.transform(target).tolist()
            label_id = [np.int32(i) for i in label_id]
        else:
            label_id = label_encoder.transform([target]).tolist()[0]
            label_id = np.int32(label_id)

        assert len(input_ids) == params.max_seq_len
        assert len(input_mask) == params.max_seq_len
        assert len(segment_ids) == params.max_seq_len, segment_ids
        if is_seq:
            assert len(label_id) == params.max_seq_len

        if ex_index < 5:
            tf.logging.debug("*** Example ***")
            tf.logging.debug("tokens: %s" % " ".join(
                [printable_text(x) for x in tokens]))
            tf.logging.debug("input_ids: %s" %
                             " ".join([str(x) for x in input_ids]))
            tf.logging.debug("input_mask: %s" %
                             " ".join([str(x) for x in input_mask]))
            tf.logging.debug("segment_ids: %s" %
                             " ".join([str(x) for x in segment_ids]))
            if is_seq or problem_type in ['seq2seq_tag', 'seq2seq_text']:
                tf.logging.debug("%s_label_ids: %s" %
                                 (problem, " ".join([str(x) for x in label_id])))
                tf.logging.debug("%s_label: %s" %
                                 (problem, " ".join([str(x) for x in target])))
            else:
                tf.logging.debug("%s_label_ids: %s" %
                                 (problem, str(label_id)))
                tf.logging.debug("%s_label: %s" %
                                 (problem, str(target)))
            if params.augument_mask_lm:
                tf.logging.debug("mask lm tokens: %s" % " ".join(
                    [printable_text(x) for x in mask_lm_tokens]))
                tf.logging.debug("mask lm input_ids: %s" %
                                 " ".join([str(x) for x in mask_lm_input_ids]))
                tf.logging.debug("mask lm label ids: %s" %
                                 " ".join([str(x) for x in masked_lm_ids]))
                tf.logging.debug("mask lm position: %s" %
                                 " ".join([str(x) for x in masked_lm_positions]))

        if not params.augument_mask_lm:
            return_dict = {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                '%s_label_ids' % problem: label_id
            }
        else:
            if random.uniform(0, 1) <= params.augument_rate:
                return_dict = {
                    'input_ids': mask_lm_input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    '%s_label_ids' % problem: label_id,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_ids": masked_lm_ids,
                    "masked_lm_weights": masked_lm_weights,
                }
            else:
                return_dict = {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    '%s_label_ids' % problem: label_id,
                    "masked_lm_positions": np.zeros_like(masked_lm_positions),
                    "masked_lm_ids": np.zeros_like(masked_lm_ids),
                    "masked_lm_weights": np.zeros_like(masked_lm_weights),
                }

        if problem_type in ['seq2seq_tag', 'seq2seq_text']:
            return_dict['%s_mask' % problem] = label_mask

        yield return_dict


def create_pretraining_generator(problem,
                                 inputs_list,
                                 target_list,
                                 label_encoder,
                                 params,
                                 tokenizer
                                 ):
    """Slight modification of original code

    Raises:
        ValueError -- Input format not right
    """

    if not isinstance(inputs_list[0][0], list):
        raise ValueError('inputs is expected to be list of list of list.')

    all_documents = []
    for document in inputs_list:
        all_documents.append([])
        for sentence in document:
            all_documents[-1].append(tokenizer.tokenize('\t'.join(sentence)))

    all_documents = [d for d in all_documents if d]
    rng = random.Random()
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []

    print_count = 0
    for _ in range(params.dupe_factor):
        for document_index in range(len(all_documents)):
            instances = create_instances_from_document(
                all_documents,
                document_index,
                params.max_seq_len,
                params.short_seq_prob,
                params.masked_lm_prob,
                params.max_predictions_per_seq,
                vocab_words, rng)
            for instance in instances:
                tokens = instance.tokens
                segment_ids = list(instance.segment_ids)

                input_mask, tokens, segment_ids, _ = create_mask_and_padding(
                    tokens, segment_ids, None, params.max_seq_len)
                masked_lm_positions = list(instance.masked_lm_positions)
                masked_lm_weights, masked_lm_labels, masked_lm_positions, _ = create_mask_and_padding(
                    instance.masked_lm_labels, masked_lm_positions, None, params.max_predictions_per_seq)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                masked_lm_ids = tokenizer.convert_tokens_to_ids(
                    masked_lm_labels)
                next_sentence_label = 1 if instance.is_random_next else 0

                yield_dict = {
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_ids": masked_lm_ids,
                    "masked_lm_weights": masked_lm_weights,
                    "next_sentence_label_ids": next_sentence_label
                }

                if print_count < 3:
                    tf.logging.debug('%s : %s' %
                                     ('tokens', ' '.join([str(x) for x in tokens])))
                    for k, v in yield_dict.items():
                        if not isinstance(v, int):
                            tf.logging.debug('%s : %s' %
                                             (k, ' '.join([str(x) for x in v])))
                    print_count += 1

                yield yield_dict


def create_generator(params, mode, epoch):
    """Function to create iterator for multiple problem

    This function dose the following things:
    1. Create dummy labels for each problems.
    2. Initialize all generators
    3. Sample a problem to train at this batch. If eval, take turns
    4. Create a loss multiplier
    5. Tried to generate samples for target problem, if failed, init gen
    6. Add dummy label to other problems

    Example:
        Problem: CWS|NER|WeiboNER&WeiboSegment
        1. Dummy labels: CWS_label_ids: [0,0,0] ...
        2. Blablabla
        3. Sample, say (WeiboNER&WeiboSegment)
        4. loss multipliers: {'CWS_loss_multiplier': 0, ..., 'WeiboNER_loss_multiplier': 1, ...}
        ...

    Arguments:
        params {Params} -- params
        mode {mode} -- mode
        epoch {int} -- epochs to run
    """
    # example
    # problem_list: ['NER', 'CWS', 'WeiboNER', 'WeiboSegment']
    # problem_chunk: [['NER'], ['CWS'], ['WeiboNER', 'WeiboSegment']]
    problem_list = []
    problem_chunk = []
    for problem_dict in params.run_problem_list:
        problem_list += list(problem_dict.keys())
        problem_chunk.append(list(problem_dict.keys()))

    # get dummy labels
    def _create_dummpy_label(problem_type):
        if problem_type == 'cls':
            return 0
        else:
            return [0]*params.max_seq_len
    dummy_label_dict = {problem+'_label_ids': _create_dummpy_label(
        params.problem_type[problem]) for problem in problem_list if params.problem_type[problem] != 'pretrain'}

    # init gen
    gen_dict = {problem: params.read_data_fn[problem](params, mode)
                for problem in problem_list}

    while gen_dict:
        # sample problem to train
        if len(problem_chunk) > 1:
            data_num_list = [params.data_num_dict[chunk[0]]
                             for chunk in problem_chunk]
            if params.multitask_balance_type == 'data_balanced':
                sample_prob = np.array(data_num_list) / np.sum(data_num_list)
                current_problem_chunk_ind = np.random.choice(
                    list(range(len(problem_chunk))), p=sample_prob)
                current_problem_chunk = problem_chunk[current_problem_chunk_ind]

            elif params.multitask_balance_type == 'problem_balanced':
                sample_prob = np.array(
                    [1]*len(data_num_list)) / np.sum([1]*len(data_num_list))
                current_problem_chunk_ind = np.random.choice(
                    list(range(len(problem_chunk))), p=sample_prob)
                current_problem_chunk = problem_chunk[current_problem_chunk_ind]
        else:
            current_problem_chunk = problem_chunk[0]

        # create loss multiplier
        loss_multiplier = {}
        for problem in problem_list:
            if problem in current_problem_chunk:
                loss_multiplier[problem+'_loss_multiplier'] = 1
            else:
                loss_multiplier[problem+'_loss_multiplier'] = 0

        base_dict = {}
        base_input = None
        for problem in current_problem_chunk:
            try:
                instance = next(gen_dict[problem])
            except StopIteration:
                if mode == 'train':
                    gen_dict[problem] = params.read_data_fn[problem](
                        params, mode)
                    instance = next(gen_dict[problem])
                else:
                    del gen_dict[problem]
                    continue
            except KeyError:
                continue

            base_dict.update(instance)
            if base_input is None:
                base_input = instance['input_ids']
            else:
                assert base_input == instance[
                    'input_ids'], 'Inputs id of two chained problem not aligned. Please double check!'

        if not base_dict:
            continue

        # add dummpy labels
        for dummy_problem in dummy_label_dict:
            if dummy_problem not in base_dict:
                base_dict[dummy_problem] = dummy_label_dict[dummy_problem]
        # add loss multipliers
        base_dict.update(loss_multiplier)
        yield base_dict


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
