import tensorflow as tf

from tensor2tensor.utils import beam_search

from .transformer_decoder import TransformerDecoder

from . import modeling
from .top_utils import (TopLayer, gather_indexes,
                        make_cudnngru, create_seq_smooth_label,
                        dense_layer)


class SequenceLabel(TopLayer):
    '''Top model for sequence labeling.
    It's a dense net with body output features as input with following support.

    crf: Conditional Random Field. Take logits(output of dense layer) as input
    hidden_gru: Take body features as input and apply rnn on it.
    label_smoothing: Hard label smoothing. Random replace label by some prob.
    '''

    def make_batch_loss(self, logits, seq_labels, seq_length, crf_transition_param):
        if self.params.crf:
            with tf.variable_scope('CRF'):
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    logits, seq_labels, seq_length,
                    transition_params=crf_transition_param)
                batch_loss = -log_likelihood
        else:
            # inconsistent shape might be introduced to labels
            # so we need to do some padding to make sure that
            # seq_labels has the same sequence length as logits
            pad_len = tf.shape(logits)[1] - tf.shape(seq_labels)[1]

            # top, bottom, left, right
            pad_tensor = [[0, 0], [0, pad_len]]
            seq_labels = tf.pad(seq_labels, paddings=pad_tensor)

            batch_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=seq_labels), axis=1)

        if self.params.uncertain_weight_loss:
            batch_loss = self.uncertainty_weighted_loss(batch_loss)
        return batch_loss

    def __call__(self, features, hidden_feature, mode, problem_name, mask=None):
        hidden_feature = hidden_feature['seq']
        scope_name = self.params.share_top[problem_name]
        if mode == tf.estimator.ModeKeys.TRAIN:
            hidden_feature = tf.nn.dropout(
                hidden_feature,
                keep_prob=self.params.dropout_keep_prob)

        if mask is None:
            num_classes = self.params.num_classes[problem_name]
        else:
            num_classes = mask.shape[0]

        # make hidden model
        hidden_feature = self.make_hidden_model(
            features, hidden_feature, mode, True)
        logits = dense_layer(num_classes, hidden_feature, mode, 1.0, None)
        self.logits = logits
        if mask is not None:
            logits = logits*mask

        # CRF transition param
        crf_transition_param = tf.get_variable(
            'crf_transition', shape=[num_classes, num_classes])

        # sequence_weight = tf.cast(features["input_mask"], tf.float32)
        seq_length = tf.reduce_sum(features["input_mask"], axis=-1)

        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_labels = features['%s_label_ids' % problem_name]
            seq_labels = create_seq_smooth_label(
                self.params, seq_labels, num_classes)
            batch_loss = self.make_batch_loss(
                logits, seq_labels, seq_length, crf_transition_param)
            self.loss = self.create_loss(
                batch_loss, features['%s_loss_multiplier' % problem_name])
            # If a batch does not contain input instances from the current problem, the loss multiplier will be empty
            # and loss will be NaN. Replacing NaN with 0 fixes the problem.
            self.loss = tf.where(tf.math.is_nan(self.loss),
                                 tf.zeros_like(self.loss), self.loss)
            return self.loss

        elif mode == tf.estimator.ModeKeys.EVAL:
            seq_labels = features['%s_label_ids' % problem_name]
            batch_loss = self.make_batch_loss(
                logits, seq_labels, seq_length, crf_transition_param)

            seq_loss = tf.reduce_mean(batch_loss)

            return self.eval_metric_fn(
                features, logits, seq_loss, problem_name, features['input_mask'], pad_labels_to_logits=True)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            if self.params.crf:
                viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
                    logits, crf_transition_param, seq_length)
                self.prob = tf.identity(
                    viterbi_sequence, name='%s_predict' % scope_name)
            else:
                self.prob = tf.nn.softmax(
                    logits, name='%s_predict' % scope_name)

            return self.prob


class Classification(TopLayer):
    '''Top model for classification.
    It's a dense net with body output features as input with following support.

    label_smoothing: Soft label smoothing.
    '''

    def create_batch_loss(self, labels, logits,  num_classes):
        if self.params.label_smoothing > 0:
            one_hot_labels = tf.one_hot(labels, depth=num_classes)
            batch_loss = tf.losses.softmax_cross_entropy(
                one_hot_labels, logits,
                label_smoothing=self.params.label_smoothing)
        else:
            batch_loss = tf.losses.sparse_softmax_cross_entropy(
                labels, logits)

        if self.params.uncertain_weight_loss:
            batch_loss = self.uncertainty_weighted_loss(batch_loss)
        return batch_loss

    def __call__(self, features, hidden_feature, mode, problem_name, mask=None):
        hidden_feature = hidden_feature['pooled']
        scope_name = self.params.share_top[problem_name]
        if mode == tf.estimator.ModeKeys.TRAIN:
            hidden_feature = tf.nn.dropout(
                hidden_feature,
                keep_prob=self.params.dropout_keep_prob)

        if mask is None:
            num_classes = self.params.num_classes.get(problem_name, 2)
        else:
            num_classes = mask.shape[0]
        # make hidden model
        hidden_feature = self.make_hidden_model(
            features, hidden_feature, mode, 'pooled')
        logits = dense_layer(num_classes, hidden_feature, mode, 1.0, None)
        self.logits = logits
        if mask is not None:
            logits = logits*mask
        if mode == tf.estimator.ModeKeys.TRAIN:
            labels = features['%s_label_ids' % problem_name]
            batch_loss = self.create_batch_loss(labels, logits, num_classes)
            self.loss = self.create_loss(
                batch_loss, features['%s_loss_multiplier' % problem_name])
            # If a batch does not contain input instances from the current problem, the loss multiplier will be empty
            # and loss will be NaN. Replacing NaN with 0 fixes the problem.
            self.loss = tf.where(tf.math.is_nan(self.loss),
                                 tf.zeros_like(self.loss), self.loss)
            return self.loss
        elif mode == tf.estimator.ModeKeys.EVAL:
            labels = features['%s_label_ids' % problem_name]
            batch_loss = self.create_batch_loss(labels, logits, num_classes)
            # multiply with loss multiplier to make some loss as zero
            loss = tf.reduce_mean(batch_loss)

            return self.eval_metric_fn(
                features, logits, loss, problem_name, pad_labels_to_logits=False)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            prob = tf.nn.softmax(logits)
            self.prob = tf.identity(prob, name='%s_predict' % scope_name)
            return self.prob


class MaskLM(TopLayer):
    '''Top model for mask language model.
    It's a dense net with body output features as input.
    Major logic is from original bert code
    '''

    def __call__(self, features, hidden_feature, mode, problem_name):
        """Get loss and log probs for the masked LM.

        DO NOT CHANGE THE VARAIBLE SCOPE.
        """
        seq_hidden_feature = hidden_feature['seq']
        positions = features['masked_lm_positions']
        input_tensor = gather_indexes(seq_hidden_feature, positions)
        output_weights = hidden_feature['embed_table']
        label_ids = features['masked_lm_ids']
        label_weights = features['masked_lm_weights']

        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=self.params.mask_lm_hidden_size,
                    activation=modeling.get_activation(
                        self.params.mask_lm_hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        self.params.mask_lm_initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[self.params.vocab_size],
                initializer=tf.zeros_initializer())

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            self.logits = logits
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            if mode == tf.estimator.ModeKeys.PREDICT:
                self.prob = log_probs
                return self.prob

            else:

                label_ids = tf.reshape(label_ids, [-1])
                label_weights = tf.reshape(label_weights, [-1])

                one_hot_labels = tf.one_hot(
                    label_ids, depth=self.params.vocab_size, dtype=tf.float32)

                # The `positions` tensor might be zero-padded (if the sequence is too
                # short to have the maximum number of predictions). The `label_weights`
                # tensor has a value of 1.0 for every real prediction and 0.0 for the
                # padding predictions.
                per_example_loss = - \
                    tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
                label_weights = tf.cast(label_weights, tf.float32)
                numerator = tf.reduce_sum(label_weights * per_example_loss)
                denominator = tf.reduce_sum(label_weights) + 1e-5
                loss = numerator / denominator

                if mode == tf.estimator.ModeKeys.TRAIN:
                    self.loss = loss
                    return self.loss

                else:
                    def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                                  masked_lm_weights):
                        """Computes the loss and accuracy of the model."""
                        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                                         [-1, masked_lm_log_probs.shape[-1]])
                        masked_lm_predictions = tf.argmax(
                            masked_lm_log_probs, axis=-1, output_type=tf.int32)
                        masked_lm_example_loss = tf.reshape(
                            masked_lm_example_loss, [-1])
                        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                        masked_lm_weights = tf.reshape(
                            masked_lm_weights, [-1])
                        masked_lm_accuracy = tf.metrics.accuracy(
                            labels=masked_lm_ids,
                            predictions=masked_lm_predictions,
                            weights=masked_lm_weights)
                        masked_lm_mean_loss = tf.metrics.mean(
                            values=masked_lm_example_loss, weights=masked_lm_weights)

                        return {
                            "masked_lm_accuracy": masked_lm_accuracy,
                            "masked_lm_loss": masked_lm_mean_loss,
                        }
                    eval_metrics = (metric_fn(
                        per_example_loss, log_probs, label_ids,
                        label_weights), loss)

                    self.eval_metrics = eval_metrics
                    return self.eval_metrics


class PreTrain(TopLayer):
    '''Top model for pretrain.
    It's MaskLM + Classification(next sentence prediction)
    '''

    def __call__(self, features, hidden_feature, mode, problem_name):
        mask_lm_top = MaskLM(self.params)
        self.params.share_top['next_sentence'] = 'next_sentence'
        mask_lm_top_result = mask_lm_top(
            features, hidden_feature, mode, problem_name)
        with tf.variable_scope('next_sentence', reuse=tf.AUTO_REUSE):
            cls = Classification(self.params)
            features['next_sentence_loss_multiplier'] = 1
            next_sentence_top_result = cls(
                features, hidden_feature, mode, 'next_sentence')
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.loss = mask_lm_top_result+next_sentence_top_result
            return self.loss
        elif mode == tf.estimator.ModeKeys.EVAL:
            mask_lm_eval_dict, mask_lm_loss = mask_lm_top_result
            next_sentence_eval_dict, next_sentence_loss = next_sentence_top_result
            mask_lm_eval_dict.update(next_sentence_eval_dict)
            self.eval_metrics = (mask_lm_eval_dict,
                                 mask_lm_loss+next_sentence_loss)
            return self.eval_metrics
        elif mode == tf.estimator.ModeKeys.PREDICT:
            self.prob = mask_lm_top_result
            return self.prob


class Seq2Seq(TopLayer):
    '''Top model for seq2seq problem.
    This is basically a decoder of encoder-decoder framework.
    Here uses transformer decoder architecture with beam search support.
    '''

    def _get_symbol_to_logit_fn(self,
                                max_seq_len,
                                embedding_table,
                                token_type_ids,
                                decoder,
                                num_classes,
                                encoder_output,
                                input_mask,
                                params):
        decoder_self_attention_mask = decoder.get_decoder_self_attention_mask(
            max_seq_len)

        batch_size = tf.shape(encoder_output)[0]
        max_seq_len = tf.shape(encoder_output)[1]

        encoder_output = tf.expand_dims(encoder_output, axis=1)
        tile_dims = [1] * encoder_output.shape.ndims
        tile_dims[1] = params.beam_size

        encoder_output = tf.tile(encoder_output, tile_dims)
        encoder_output = tf.reshape(encoder_output,
                                    [-1, max_seq_len, params.bert_config.hidden_size])

        def symbols_to_logits_fn(ids, i, cache):

            decoder_inputs = tf.nn.embedding_lookup(
                embedding_table, ids)

            decoder_inputs = modeling.embedding_postprocessor(
                input_tensor=decoder_inputs,
                use_token_type=False,
                token_type_ids=token_type_ids,
                token_type_vocab_size=params.bert_config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=params.bert_config.initializer_range,
                max_position_embeddings=params.bert_config.max_position_embeddings,
                dropout_prob=self.params.bert_config.hidden_dropout_prob)
            final_decoder_input = decoder_inputs[:, -1:, :]
            # final_decoder_input = decoder_inputs
            self_attention_mask = decoder_self_attention_mask[:, i:i+1, :i+1]

            logits = decoder.decode(
                decoder_inputs=final_decoder_input,
                encoder_output=encoder_output,
                input_mask=input_mask,
                decoder_self_attention_mask=self_attention_mask,
                cache=cache,
                num_classes=num_classes,
                do_return_all_layers=False)

            return logits, cache
        return symbols_to_logits_fn

    def beam_search_decode(self, features, hidden_feature, mode, problem_name):
        # prepare inputs to attention
        key = 'ori_seq' if self.params.label_transfer else 'seq'
        encoder_outputs = hidden_feature[key]
        max_seq_len = self.params.max_seq_len
        embedding_table = hidden_feature['embed_table']
        token_type_ids = features['segment_ids']
        num_classes = self.params.num_classes[problem_name]
        batch_size = modeling.get_shape_list(
            encoder_outputs, expected_rank=3)[0]
        hidden_size = self.params.bert_config.hidden_size

        if self.params.problem_type[problem_name] == 'seq2seq_text':
            embedding_table = hidden_feature['embed_table']
        else:
            embedding_table = tf.get_variable(
                'tag_embed_table',
                shape=[num_classes, hidden_size])

        symbol_to_logit_fn = self._get_symbol_to_logit_fn(
            max_seq_len=max_seq_len,
            embedding_table=embedding_table,
            token_type_ids=token_type_ids,
            decoder=self.decoder,
            num_classes=num_classes,
            encoder_output=encoder_outputs,
            input_mask=features['input_mask'],
            params=self.params
        )

        # create cache for fast decode
        cache = {
            str(layer): {
                "key_layer": tf.zeros([batch_size, 0, hidden_size]),
                "value_layer": tf.zeros([batch_size, 0, hidden_size]),
            } for layer in range(self.params.decoder_num_hidden_layers)}
        # cache['encoder_outputs'] = encoder_outputs
        # cache['encoder_decoder_attention_mask'] = features['input_mask']
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        decode_ids, _, _ = beam_search.beam_search(
            symbols_to_logits_fn=symbol_to_logit_fn,
            initial_ids=initial_ids,
            states=cache,
            vocab_size=self.params.num_classes[problem_name],
            beam_size=self.params.beam_size,
            alpha=self.params.beam_search_alpha,
            decode_length=self.params.decode_max_seq_len,
            eos_id=self.params.eos_id[problem_name])
        # Get the top sequence for each batch element
        top_decoded_ids = decode_ids[:, 0, 1:]
        self.prob = top_decoded_ids
        return self.prob

    def __call__(self, features, hidden_feature, mode, problem_name):
        self.decoder = TransformerDecoder(self.params)
        scope_name = self.params.share_top[problem_name]

        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = features['%s_label_ids' % problem_name]

            logits = self.decoder.train_eval(
                features, hidden_feature, mode, problem_name)

            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                shift_labels = tf.pad(
                    labels, [[0, 0], [0, 1]])[:, 1:]
            batch_loss = tf.losses.sparse_softmax_cross_entropy(
                shift_labels, logits)
            loss = self.create_loss(
                batch_loss, features['%s_loss_multiplier' % problem_name])
            # If a batch does not contain input instances from the current problem, the loss multiplier will be empty
            # and loss will be NaN. Replacing NaN with 0 fixes the problem.
            loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
            self.loss = loss

            if mode == tf.estimator.ModeKeys.TRAIN:
                return self.loss
            else:
                return self.eval_metric_fn(
                    features, logits, loss, problem_name, features['%s_mask' % problem_name])

        else:
            self.pred = tf.identity(self.beam_search_decode(
                features, hidden_feature, mode, problem_name),
                name='%s_predict' % scope_name)
            return self.pred


class MultiLabelClassification(TopLayer):
    '''Top model for multi-class classification.
    It's a dense net with body output features as input with following support.

    label_smoothing: Soft label smoothing.
    '''

    def create_batch_loss(self, labels, logits,  num_classes):
        labels = tf.cast(labels, tf.float32)
        batch_label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)

        batch_loss = tf.reduce_sum(batch_label_loss, axis=1)

        if self.params.uncertain_weight_loss:
            batch_loss = self.uncertainty_weighted_loss(batch_loss)
        return batch_loss

    def __call__(self, features, hidden_feature, mode, problem_name, mask=None):
        hidden_feature = hidden_feature['pooled']
        scope_name = self.params.share_top[problem_name]
        if mode == tf.estimator.ModeKeys.TRAIN:
            hidden_feature = tf.nn.dropout(
                hidden_feature,
                keep_prob=self.params.dropout_keep_prob)

        if mask is None:
            num_classes = self.params.num_classes[problem_name]
        else:
            num_classes = mask.shape[0]
        # make hidden model
        hidden_feature = self.make_hidden_model(
            features, hidden_feature, mode, 'pooled')
        logits = dense_layer(num_classes, hidden_feature, mode, 1.0, None)
        self.logits = logits
        if mask is not None:
            logits = logits*mask
        if mode == tf.estimator.ModeKeys.TRAIN:
            labels = features['%s_label_ids' % problem_name]
            batch_loss = self.create_batch_loss(labels, logits, num_classes)
            self.loss = self.create_loss(
                batch_loss, features['%s_loss_multiplier' % problem_name])
            # If a batch does not contain input instances from the current problem, the loss multiplier will be empty
            # and loss will be NaN. Replacing NaN with 0 fixes the problem.
            self.loss = tf.where(tf.math.is_nan(self.loss),
                                 tf.zeros_like(self.loss), self.loss)
            return self.loss
        elif mode == tf.estimator.ModeKeys.EVAL:
            labels = features['%s_label_ids' % problem_name]
            batch_loss = self.create_batch_loss(labels, logits, num_classes)
            # multiply with loss multiplier to make some loss as zero
            loss = tf.reduce_mean(batch_loss)
            prob = tf.nn.sigmoid(logits)
            prob = tf.round(prob)
            prob = tf.expand_dims(prob, -1)
            return self.eval_metric_fn(
                features, prob, loss, problem_name)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            prob = tf.nn.sigmoid(logits)
            self.prob = tf.identity(prob, name='%s_predict' % scope_name)
            return self.prob
