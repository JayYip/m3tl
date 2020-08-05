import tensorflow as tf
from copy import copy

from .transformer_decoder import TransformerDecoder

from . import modeling
from .top_utils import (TopLayer, gather_indexes,
                        make_cudnngru, create_seq_smooth_label,
                        dense_layer)
from .top import SequenceLabel, Classification


class LabelTransferHidden(TopLayer):
    '''Top model for label transfer, specific for multitask.
    It's a dense net with body output features as input.

    This layer will apply SequenceLabel or Classification for each problem
    and then concat each problems' logits together as new hidden feature.

    '''

    def __call__(self, features, hidden_feature, mode):
        new_hidden_feature = {}
        seq_hidden_state = []
        pooled_hidden_state = []

        if self.params.hidden_gru:
            hidden_gru = True
        else:
            hidden_gru = False

        self.params.hidden_gru = False

        for problem_dict in self.params.train_problem:
            for problem in problem_dict:
                scope_name = self.params.share_top[problem]

                top_scope_name = '%s_top' % scope_name

                with tf.variable_scope(top_scope_name, reuse=tf.AUTO_REUSE):
                    if self.params.problem_type[problem] == 'seq_tag':
                        seq_tag = SequenceLabel(self.params)
                        seq_tag(features,
                                hidden_feature,
                                tf.estimator.ModeKeys.PREDICT,
                                problem)

                        seq_hidden_state.append(seq_tag.get_logit())
                    elif self.params.problem_type[problem] == 'cls':
                        cls = Classification(self.params)

                        cls(features,
                            hidden_feature,
                            tf.estimator.ModeKeys.PREDICT,
                            problem)
                        pooled_hidden_state.append(cls.get_logit())

        self.params.hidden_gru = hidden_gru

        if len(seq_hidden_state) >= 2:
            new_hidden_feature['seq'] = tf.concat(seq_hidden_state, axis=-1)
        if len(pooled_hidden_state) >= 2:
            new_hidden_feature['pooled'] = tf.concat(
                pooled_hidden_state, axis=-1)
        hidden_feature.update(new_hidden_feature)

        if self.params.label_transfer_gru:

            lt_hidden_size = 0
            for problem_dict in self.params.train_problem:
                for p in problem_dict:
                    lt_hidden_size += self.params.num_classes[p]

            seq_features = hidden_feature['seq']
            if self.params.label_transfer_gru_hidden_size is not None:
                lt_hidden_size = self.params.label_transfer_gru_hidden_size

            input_hidden_size = seq_features.get_shape().as_list()[-1]
            with tf.variable_scope('label_transfer_rnn'):
                rnn_output = make_cudnngru(
                    seq_features,
                    lt_hidden_size,
                    self.params,
                    mode,
                    True,
                    'ave')
                rnn_output.set_shape(
                    [None, self.params.max_seq_len, input_hidden_size])
            hidden_feature['seq'] = rnn_output

        return hidden_feature


def create_multiself_attention_mask(
        problem_type,
        query_hidden_feature,
        input_mask,
        input_ids,
        duplicate_factor):
    '''A simple function to create multiself attention mask.

    This is kinda a kinda combination of self attention mask and
    encoder decoder attention mask. 

    Like self attention mask, it can 'see' all words. Unlike self
    attention, it has encoder decoder mask which is just a stack 
    of self attention.

    Arguments:
        problem_type {str} -- problem type
        query_hidden_feature {tensor} -- query
        input_mask {tensor} -- input mask
        input_ids {tensor} -- input ids
        duplicate_factor {int} -- number of self attention to stack

    Returns:
        tuple -- self_attention_mask, enc_dec_attention_mask, self_attention
    '''

    if problem_type == 'cls':
        self_attention_mask = modeling.create_attention_mask_from_input_mask(
            query_hidden_feature, input_mask)
        enc_dec_attention_mask = tf.concat(
            [self_attention_mask]*duplicate_factor, axis=-1)
        self_attention_mask = self_attention_mask[:, :, 0]
        # For cls, no need to add self attention
        self_attention = False
    else:
        self_attention_mask = modeling.create_attention_mask_from_input_mask(
            input_ids, input_mask)
        enc_dec_attention_mask = tf.concat(
            [self_attention_mask]*duplicate_factor, axis=-1)
        self_attention = True

    return (self_attention_mask, enc_dec_attention_mask, self_attention)


class GridTransformer(TopLayer):

    def __call__(self, features, hidden_feature, mode, problem_name):
        key_hidden_feature = hidden_feature['all']

        problem_type = self.params.problem_type[problem_name]
        q_key = 'pooled' if problem_type == 'cls' else 'seq'
        hidden_feature[q_key] = self.make_hidden_model(
            features, hidden_feature[q_key], mode, q_key == 'seq')
        query_hidden_feature = hidden_feature[q_key]
        if problem_type == 'cls':
            query_hidden_feature = tf.expand_dims(
                query_hidden_feature, axis=1)
        hidden_size = self.params.bert_config.hidden_size

        # transform hidden feature to batch_size, max_seq*num_layers, hidden_size
        key_hidden_feature = tf.reshape(
            key_hidden_feature,
            [-1, self.params.bert_config.num_hidden_layers*self.params.max_seq_len, hidden_size])

        grid_transformer_params = copy(self.params)
        grid_transformer_params.decoder_num_hidden_layers = 1
        grid_transformer_params.decode_max_seq_len = self.params.max_seq_len
        self.decoder = TransformerDecoder(grid_transformer_params)

        encoder_output = key_hidden_feature
        decoder_inputs = query_hidden_feature
        input_mask = features['input_mask']

        self_attention_mask, enc_dec_attention_mask, self_attention = create_multiself_attention_mask(
            problem_type,
            query_hidden_feature,
            input_mask,
            features['input_ids'],
            grid_transformer_params.bert_num_hidden_layer
        )

        decode_output = self.decoder.decode(
            decoder_inputs=decoder_inputs,
            encoder_output=encoder_output,
            input_mask=input_mask,
            decoder_self_attention_mask=self_attention_mask,
            cache=None,
            num_classes=None,
            do_return_all_layers=False,
            enc_dec_attention_mask=enc_dec_attention_mask,
            add_self_attention=self_attention
        )
        if problem_type == 'cls':
            decode_output = tf.squeeze(decode_output, axis=1)

        return decode_output


class TaskTransformer(TopLayer):
    '''Top model for label transfer, specific for multitask.
    It's a dense net with body output features as input.

    This layer will apply SequenceLabel or Classification for each problem
    and then concat each problems' logits together as new hidden feature.

    '''

    def __call__(self, features, hidden_feature, mode):

        self.params.hidden_gru = False
        self.params.hidden_dense = True

        # intermedian dense
        hidden_logits = {}
        for problem_dict in self.params.train_problem:
            for problem in problem_dict:
                scope_name = self.params.share_top[problem]

                top_scope_name = '%s_top' % scope_name
                with tf.variable_scope(top_scope_name, reuse=tf.AUTO_REUSE):

                    if self.params.problem_type[problem] == 'seq_tag':
                        seq_tag = SequenceLabel(self.params)
                        seq_tag(features,
                                hidden_feature,
                                mode,
                                problem)
                    elif self.params.problem_type[problem] == 'cls':
                        cls = Classification(self.params)

                        cls(features,
                            hidden_feature,
                            mode,
                            problem)
                    hidden_logits[problem] = seq_tag.hidden_model_logit

                    # hidden_logits[problem] = self.make_hidden_model(
                    #     features, hidden_feature[q_key], mode, q_key == 'seq')

        # task attention
        task_attention_logits = {}
        transformer_params = copy(self.params)
        transformer_params.decoder_num_hidden_layers = 1
        transformer_params.decode_max_seq_len = self.params.max_seq_len
        for problem_dict in self.params.train_problem:
            for problem in problem_dict:
                scope_name = self.params.share_top[problem]
                problem_type = self.params.problem_type[problem]

                top_scope_name = '%s_top_task_attention' % scope_name
                with tf.variable_scope(top_scope_name, reuse=tf.AUTO_REUSE):
                    with tf.variable_scope('task_attention'):
                        other_task_logits = tf.concat([
                            v for k, v in hidden_logits.items()],
                            axis=1)
                        encoder_output = other_task_logits
                        decoder_inputs = hidden_logits[problem]
                        input_mask = features['input_mask']
                        self_attention_mask, enc_dec_attention_mask, self_attention = create_multiself_attention_mask(
                            problem_type,
                            decoder_inputs,
                            input_mask,
                            features['input_ids'],
                            len(hidden_logits)
                        )
                        decoder = TransformerDecoder(transformer_params)
                        decode_output = decoder.decode(
                            decoder_inputs=decoder_inputs,
                            encoder_output=encoder_output,
                            input_mask=input_mask,
                            decoder_self_attention_mask=self_attention_mask,
                            cache=None,
                            num_classes=None,
                            do_return_all_layers=False,
                            enc_dec_attention_mask=enc_dec_attention_mask,
                            add_self_attention=self_attention
                        )
                        if problem_type == 'cls':
                            task_attention_logits[problem] = {
                                'pooled': tf.concat([decode_output, hidden_feature['pooled']], axis=-1)}
                        else:
                            task_attention_logits[problem] = {
                                'seq': tf.concat([decode_output, hidden_feature['seq']], axis=-1)
                            }

        return task_attention_logits
