import os
import random
import tempfile
from copy import copy
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
import pickle

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from ..special_tokens import BOS_TOKEN, EOS_TOKEN

from .tokenization import printable_text
from .bert_utils import (add_special_tokens_with_seqs,
                         create_instances_from_document, create_mask_and_padding,
                         create_masked_lm_predictions, punc_augument,
                         tokenize_text_with_seqs, truncate_seq_pair)


def create_bert_features(problem,
                         example_list,
                         label_encoder,
                         params,
                         tokenizer,
                         mode,
                         problem_type,
                         is_seq):
    if params.get_problem_type(problem) == 'pretrain':
        return create_bert_pretraining(
            problem=problem,
            inputs_list=example_list,
            label_encoder=label_encoder,
            params=params,
            tokenizer=tokenizer
        )

    return_dict_list = []
    for example in example_list:
        raw_inputs, raw_target = example

        # punctuation augumentation
        if params.punc_replace_prob > 0 and mode == 'train':
            raw_inputs = punc_augument(raw_inputs, params)

        # tokenize inputs, now the length is fixed, target == raw_target
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
        # check whether tokenization changed the length
        if is_seq:
            if len(target) != len(tokens_a):
                continue

        # truncate tokens and target to max_seq_len
        tokens_a, tokens_b, target = truncate_seq_pair(
            tokens_a, tokens_b, target, params.max_seq_len, is_seq=is_seq)

        # add [SEP], [CLS] tokens
        tokens, segment_ids, target = add_special_tokens_with_seqs(
            tokens_a, tokens_b, target, is_seq)

        # train mask lm as augument task while training
        if params.augument_mask_lm and mode == 'train':
            rng = random.Random()
            (mask_lm_tokens, masked_lm_positions,
                masked_lm_labels) = create_masked_lm_predictions(
                    tokens,
                    params.masked_lm_prob,
                    params.max_predictions_per_seq,
                    list(tokenizer.vocab.keys()), rng)
            _, mask_lm_tokens, _, _ = create_mask_and_padding(
                mask_lm_tokens,
                copy(segment_ids),
                copy(target),
                params.max_seq_len,
                is_seq,
                dynamic_padding=params.dynamic_padding)
            masked_lm_weights, masked_lm_labels, masked_lm_positions, _ = create_mask_and_padding(
                masked_lm_labels, masked_lm_positions, None, params.max_predictions_per_seq)
            mask_lm_input_ids = tokenizer.convert_tokens_to_ids(
                mask_lm_tokens)
            masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_mask, tokens, segment_ids, target = create_mask_and_padding(
            tokens, segment_ids, target, params.max_seq_len, is_seq, dynamic_padding=params.dynamic_padding)

        # create mask and padding for labels of seq2seq problem
        if problem_type in ['seq2seq_tag', 'seq2seq_text']:

            # tokenize text if target is text
            if problem_type == 'seq2seq_text':

                # assign num_classes for text generation problem
                params.num_classes[problem] = len(label_encoder.vocab)

                target, _ = tokenize_text_with_seqs(
                    label_encoder, target, None, False)

            target, _, _ = truncate_seq_pair(
                target, None, None, params.decode_max_seq_len, is_seq=is_seq)
            # since we initialize the id to 0 in prediction, we need
            # to make sure that BOS_TOKEN is [PAD]
            target = [BOS_TOKEN] + target + [EOS_TOKEN]
            label_mask, target, _, _ = create_mask_and_padding(
                target, [0] * len(target), None, params.decode_max_seq_len)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if isinstance(target, list):
            if problem_type == 'seq2seq_text':
                label_id = label_encoder.convert_tokens_to_ids(target)
            elif problem_type == 'multi_cls':
                label_id = label_encoder.transform([target])[0]
            else:
                # seq2seq_tag
                label_id = label_encoder.transform(target).tolist()
                label_id = [np.int32(i) for i in label_id]
        else:
            label_id = label_encoder.transform([target]).tolist()[0]
            label_id = np.int32(label_id)

        if not params.dynamic_padding:
            assert len(input_ids) == params.max_seq_len
            assert len(input_mask) == params.max_seq_len
            assert len(segment_ids) == params.max_seq_len, segment_ids
            if is_seq:
                assert len(label_id) == params.max_seq_len

        # create return dict
        if not params.augument_mask_lm:
            return_dict = {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                '%s_label_ids' % problem: label_id
            }
        else:
            if mode == 'train' and random.uniform(0, 1) <= params.augument_rate:
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
                    "masked_lm_positions": np.zeros([params.max_predictions_per_seq]),
                    "masked_lm_ids": np.zeros([params.max_predictions_per_seq]),
                    "masked_lm_weights": np.zeros([params.max_predictions_per_seq]),
                }

        if problem_type in ['seq2seq_tag', 'seq2seq_text']:
            return_dict['%s_mask' % problem] = label_mask

        return_dict_list.append(return_dict)

    return return_dict_list


def create_bert_pretraining(problem,
                            inputs_list,
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
    return_list = []
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

                return_list.append(yield_dict)
    return return_list
