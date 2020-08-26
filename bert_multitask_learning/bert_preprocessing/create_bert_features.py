import logging
import random
from copy import copy

import numpy as np
import tensorflow as tf

from ..special_tokens import PREDICT
from .bert_utils import (create_instances_from_document,
                         create_masked_lm_predictions)

from transformers import PreTrainedTokenizer


def seq_tag_label_handling(tokenized_dict, target, pad_token):
    special_token_mask = tokenized_dict['special_tokens_mask']
    del tokenized_dict['special_tokens_mask']
    processed_target = []
    for m in special_token_mask:
        # 0 is special tokens, 1 is tokens
        if m == 1:
            # add pad
            processed_target.append(pad_token)
        else:
            processed_target.append(target.pop(0))
    return processed_target, tokenized_dict


def pad_wrapper(inp, target_len=90):
    if len(inp) >= target_len:
        return inp[:target_len]
    else:
        return inp + [0]*(target_len - len(inp))


def convert_labels_to_ids(target, problem_type, label_encoder, tokenizer=None, decoding_length=None):
    label_mask = None
    if isinstance(target, list):
        if problem_type == 'seq2seq_text':

            target = [label_encoder.bos_token] + \
                target + [label_encoder.eos_token]
            label_dict = label_encoder(
                target, add_special_tokens=False, is_pretokenized=True)
            label_id = label_dict['input_ids']
            label_mask = [0] + label_dict['attention_mask'] + [0]
            label_id = pad_wrapper(label_id, decoding_length)
            label_mask = pad_wrapper(label_mask, decoding_length)

        elif problem_type == 'multi_cls':
            label_id = label_encoder.transform([target])[0]
        elif problem_type == 'seq2seq_tag':
            # seq2seq_tag
            target = [label_encoder.bos_token] + \
                target + [label_encoder.eos_token]
            label_dict = tokenizer(
                target, is_pretokenized=True, add_special_tokens=False)
            label_mask = [0] + label_dict['attention_mask'] + [0]
            label_id = label_encoder.transform(target).tolist()
            label_id = [np.int32(i) for i in label_id]
        else:
            label_id = label_encoder.transform(target).tolist()
            label_id = [np.int32(i) for i in label_id]
    else:
        if problem_type == 'seq2seq_text':
            target = label_encoder.bos_token + target + label_encoder.eos_token
            label_dict = label_encoder(
                target, add_special_tokens=False, is_pretokenized=False)
            label_id = label_dict['input_ids']
            label_mask = [0] + label_dict['attention_mask'] + [0]
            label_id = pad_wrapper(label_id, decoding_length)
            label_mask = pad_wrapper(label_mask, decoding_length)
        else:
            label_id = label_encoder.transform([target]).tolist()[0]
            label_id = np.int32(label_id)
    return label_id, label_mask


def _create_bert_features(problem,
                          example_list,
                          label_encoder,
                          params,
                          tokenizer: PreTrainedTokenizer,
                          mode,
                          problem_type,
                          is_seq):

    for example in example_list:
        try:
            raw_inputs, raw_target = example
        except ValueError:
            raw_inputs = example
            raw_target = None

        # # tokenize inputs, now the length is fixed, target == raw_target
        if isinstance(raw_inputs, dict):
            tokens_a = raw_inputs['a']
            tokens_b = raw_inputs['b']
        else:
            tokens_a = raw_inputs
            tokens_b = None

        target = raw_target

        if isinstance(tokens_a, list):
            is_pretokenized = True
        else:
            is_pretokenized = False

        tokenized_dict = tokenizer(
            tokens_a, tokens_b,
            truncation=False,
            is_pretokenized=is_pretokenized,
            padding=False,
            return_special_tokens_mask=is_seq,
            add_special_tokens=True)

        # check whether tokenization changed the length
        if is_seq:
            target, tokenized_dict = seq_tag_label_handling(
                tokenized_dict, target, tokenizer.pad_token)

            if len(target) != len(tokenized_dict['input_ids']):
                raise ValueError(
                    'Length is different for seq tag problem, inputs: {}'.format(tokenizer.decode(tokenized_dict['input_ids'])))

        if mode != PREDICT:

            label_id, label_mask = convert_labels_to_ids(
                target, problem_type, label_encoder, tokenizer, params.decode_max_seq_len)

        input_ids = tokenized_dict['input_ids']
        segment_ids = tokenized_dict['token_type_ids']
        input_mask = tokenized_dict['attention_mask']

        # create return dict
        if mode != PREDICT:
            return_dict = {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                '%s_label_ids' % problem: label_id
            }
        else:
            return_dict = {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
        if problem_type in ['seq2seq_tag', 'seq2seq_text']:
            return_dict['%s_mask' % problem] = label_mask
        yield return_dict


def create_bert_features(problem,
                         example_list,
                         label_encoder,
                         params,
                         tokenizer,
                         mode,
                         problem_type,
                         is_seq):
    if problem_type == 'pretrain':
        return create_bert_pretraining(
            problem=problem,
            inputs_list=example_list,
            label_encoder=label_encoder,
            params=params,
            tokenizer=tokenizer
        )
    gen = _create_bert_features(problem,
                                example_list,
                                label_encoder,
                                params,
                                tokenizer,
                                mode,
                                problem_type,
                                is_seq)
    return_dict_list = [d for d in gen]
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

                masked_lm_positions = list(instance.masked_lm_positions)

                next_sentence_label = 1 if instance.is_random_next else 0

                mask_lm_dict = tokenizer(instance.masked_lm_labels,
                                         truncation=False,
                                         is_pretokenized=True,
                                         padding='max_length',
                                         max_length=params.max_predictions_per_seq,
                                         return_special_tokens_mask=False,
                                         add_special_tokens=False)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1 for _ in input_ids]
                masked_lm_ids = mask_lm_dict['input_ids']
                masked_lm_weights = mask_lm_dict['attention_mask']
                masked_lm_positions = masked_lm_positions + \
                    masked_lm_ids[len(masked_lm_positions):]

                assert len(input_ids) == len(
                    segment_ids), (len(input_ids), len(segment_ids))
                assert len(masked_lm_ids) == len(masked_lm_positions), (len(
                    masked_lm_ids), len(masked_lm_positions))

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
                    tf.compat.v1.logging.debug('%s : %s' %
                                               ('tokens', ' '.join([str(x) for x in tokens])))
                    for k, v in yield_dict.items():
                        if not isinstance(v, int):
                            tf.compat.v1.logging.debug('%s : %s' %
                                                       (k, ' '.join([str(x) for x in v])))
                    print_count += 1

                return_list.append(yield_dict)
    return return_list


def _create_multimodal_bert_features(problem,
                                     example_list,
                                     label_encoder,
                                     params,
                                     tokenizer,
                                     mode,
                                     problem_type,
                                     is_seq):
    if problem_type == 'pretrain':
        raise NotImplementedError('Multimodal Pretraining is not implemented')

    for example in example_list:
        try:
            raw_inputs, raw_target = example
        except ValueError:
            raw_inputs = example
            raw_target = None

        if problem_type == 'seq_tag' and not isinstance(raw_target, dict):
            raise ValueError(
                'Label of multimodal sequence tagging must be a dictionary')

        if not isinstance(raw_inputs, dict):
            raise ValueError(
                'Multimodal inputs is supposed to be a dictionary')

        if isinstance(raw_target, dict):
            target_by_modal = True
        else:
            target_by_modal = False

        modal_name_list = ['text', 'image', 'others']

        return_dict = {}
        for modal_name in modal_name_list:
            if modal_name not in raw_inputs:
                continue

            modal_inputs = raw_inputs[modal_name]

            if target_by_modal:
                modal_target = raw_target[modal_name]
            else:
                modal_target = raw_target

            if modal_name == 'text':
                # tokenize inputs, now the length is fixed, target == raw_target
                if isinstance(modal_inputs, dict):
                    tokens_a = modal_inputs['a']
                    tokens_b = modal_inputs['b']
                else:
                    tokens_a = modal_inputs
                    tokens_b = None
                target = modal_target

                if isinstance(tokens_a, list):
                    is_pretokenized = True
                else:
                    is_pretokenized = False
                tokenized_dict = tokenizer(
                    tokens_a, tokens_b,
                    truncation=False,
                    is_pretokenized=is_pretokenized,
                    padding=False,
                    return_special_tokens_mask=is_seq,
                    add_special_tokens=True)

                if is_seq:
                    target, tokenized_dict = seq_tag_label_handling(
                        tokenized_dict, target, tokenizer.pad_token)

                    if len(target) != len(tokenized_dict['input_ids']):
                        raise ValueError(
                            'Length is different for seq tag problem, inputs: {}'.format(tokenizer.decode(tokenized_dict['input_ids'])))

                input_ids = tokenized_dict['input_ids']
                segment_ids = tokenized_dict['token_type_ids']
                input_mask = tokenized_dict['attention_mask']

                modal_feature_dict = {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids
                }

            else:
                modal_inputs = np.array(modal_inputs)
                if len(modal_inputs.shape) == 1:
                    modal_inputs = np.expand_dims(modal_inputs, axis=0)
                target = modal_target
                segment_ids = np.zeros(
                    modal_inputs.shape[0], dtype=np.int32) + params.modal_token_type_id[modal_name]
                input_mask = [1]*len(modal_inputs)
                modal_feature_dict = {
                    '{}_input'.format(modal_name): modal_inputs,
                    '{}_mask'.format(modal_name): input_mask,
                    '{}_segment_ids'.format(modal_name): segment_ids}

            # encode labels
            if mode != PREDICT:
                label_id, label_mask = convert_labels_to_ids(
                    target, problem_type, label_encoder, tokenizer, params.decode_max_seq_len)

                if target_by_modal:
                    modal_feature_dict['{}_{}_label_ids'.format(
                        problem, modal_name)] = label_id
                else:
                    modal_feature_dict['{}_label_ids'.format(
                        problem)] = label_id
            return_dict.update(modal_feature_dict)

        if problem_type in ['seq2seq_tag', 'seq2seq_text']:
            return_dict['%s_mask' % problem] = label_mask
        yield return_dict


def create_multimodal_bert_features(problem,
                                    example_list,
                                    label_encoder,
                                    params,
                                    tokenizer,
                                    mode,
                                    problem_type,
                                    is_seq):
    if problem_type == 'pretrain':
        raise NotImplementedError("Multimodal pretraining is not implemented")
    gen = _create_multimodal_bert_features(problem,
                                           example_list,
                                           label_encoder,
                                           params,
                                           tokenizer,
                                           mode,
                                           problem_type,
                                           is_seq)
    return_dict_list = [d for d in gen]
    return return_dict_list


def create_bert_features_generator(problem,
                                   example_list,
                                   label_encoder,
                                   params,
                                   tokenizer,
                                   mode,
                                   problem_type,
                                   is_seq):
    if problem_type == 'pretrain':
        raise ValueError('pretraining does not support generator')
    gen = _create_bert_features(problem,
                                example_list,
                                label_encoder,
                                params,
                                tokenizer,
                                mode,
                                problem_type,
                                is_seq)
    return gen


def create_multimodal_bert_features_generator(problem,
                                              example_list,
                                              label_encoder,
                                              params,
                                              tokenizer,
                                              mode,
                                              problem_type,
                                              is_seq):
    if problem_type == 'pretrain':
        raise ValueError('pretraining does not support generator')
    gen = _create_multimodal_bert_features(problem,
                                           example_list,
                                           label_encoder,
                                           params,
                                           tokenizer,
                                           mode,
                                           problem_type,
                                           is_seq)
    return gen
