import tensorflow as tf

from . import modeling
from .top_utils import (TopLayer, gather_indexes,
                        create_seq_smooth_label,
                        dense_layer)
from .utils import load_transformer_model


class SequenceLabel(TopLayer):
    # pylint: disable=attribute-defined-outside-init
    '''Top model for sequence labeling.
    It's a dense net with body output features as input with following support.

    crf: Conditional Random Field. Take logits(output of dense layer) as input
    hidden_gru: Take body features as input and apply rnn on it.
    label_smoothing: Hard label smoothing. Random replace label by some prob.
    '''

    def make_batch_loss(self, logits, seq_labels, seq_length, crf_transition_param):
        if self.params.crf:
            with tf.compat.v1.variable_scope('CRF'):
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    logits, seq_labels, seq_length,
                    transition_params=crf_transition_param)
                batch_loss = -log_likelihood
        else:
            # inconsistent shape might be introduced to labels
            # so we need to do some padding to make sure that
            # seq_labels has the same sequence length as logits
            pad_len = tf.shape(input=logits)[1] - tf.shape(input=seq_labels)[1]

            # top, bottom, left, right
            pad_tensor = [[0, 0], [0, pad_len]]
            seq_labels = tf.pad(tensor=seq_labels, paddings=pad_tensor)

            batch_loss = tf.reduce_mean(
                input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
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
                rate=1 - (self.params.dropout_keep_prob))

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
        crf_transition_param = tf.compat.v1.get_variable(
            'crf_transition', shape=[num_classes, num_classes])

        # sequence_weight = tf.cast(features["input_mask"], tf.float32)
        seq_length = tf.reduce_sum(
            input_tensor=features["input_mask"], axis=-1)

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
            self.loss = tf.compat.v1.where(tf.math.is_nan(self.loss),
                                           tf.zeros_like(self.loss), self.loss)
            return self.loss

        elif mode == tf.estimator.ModeKeys.EVAL:
            seq_labels = features['%s_label_ids' % problem_name]
            batch_loss = self.make_batch_loss(
                logits, seq_labels, seq_length, crf_transition_param)

            seq_loss = tf.reduce_mean(input_tensor=batch_loss)

            return self.eval_metric_fn(
                features, logits, seq_loss, problem_name, features['input_mask'], pad_labels_to_logits=True)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            if self.params.crf:
                viterbi_sequence, _ = tf.contrib.crf.crf_decode(
                    logits, crf_transition_param, seq_length)
                self.prob = tf.identity(
                    viterbi_sequence, name='%s_predict' % scope_name)
            else:
                self.prob = tf.nn.softmax(
                    logits, name='%s_predict' % scope_name)

            return self.prob


class Classification(TopLayer):
    # pylint: disable=attribute-defined-outside-init
    '''Top model for classification.
    It's a dense net with body output features as input with following support.

    label_smoothing: Soft label smoothing.
    '''

    def create_batch_loss(self, labels, logits,  num_classes):
        if self.params.label_smoothing > 0:
            one_hot_labels = tf.one_hot(labels, depth=num_classes)
            batch_loss = tf.compat.v1.losses.softmax_cross_entropy(
                one_hot_labels, logits,
                label_smoothing=self.params.label_smoothing)
        else:
            batch_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
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
                rate=1 - (self.params.dropout_keep_prob))

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
            self.loss = tf.compat.v1.where(tf.math.is_nan(self.loss),
                                           tf.zeros_like(self.loss), self.loss)
            return self.loss
        elif mode == tf.estimator.ModeKeys.EVAL:
            labels = features['%s_label_ids' % problem_name]
            batch_loss = self.create_batch_loss(labels, logits, num_classes)
            # multiply with loss multiplier to make some loss as zero
            loss = tf.reduce_mean(input_tensor=batch_loss)

            return self.eval_metric_fn(
                features, logits, loss, problem_name, pad_labels_to_logits=False)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            prob = tf.nn.softmax(logits)
            self.prob = tf.identity(prob, name='%s_predict' % scope_name)
            return self.prob


class MaskLM(TopLayer):
    # pylint: disable=attribute-defined-outside-init
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

        with tf.compat.v1.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.compat.v1.variable_scope("transform"):
                input_tensor = tf.compat.v1.layers.dense(
                    input_tensor,
                    units=self.params.mask_lm_hidden_size,
                    activation=modeling.get_activation(
                        self.params.mask_lm_hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        self.params.mask_lm_initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.compat.v1.get_variable(
                "output_bias",
                shape=[self.params.vocab_size],
                initializer=tf.compat.v1.zeros_initializer())

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
                    tf.reduce_sum(input_tensor=log_probs *
                                  one_hot_labels, axis=[-1])
                label_weights = tf.cast(label_weights, tf.float32)
                numerator = tf.reduce_sum(
                    input_tensor=label_weights * per_example_loss)
                denominator = tf.reduce_sum(input_tensor=label_weights) + 1e-5
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
                            input=masked_lm_log_probs, axis=-1, output_type=tf.int32)
                        masked_lm_example_loss = tf.reshape(
                            masked_lm_example_loss, [-1])
                        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                        masked_lm_weights = tf.reshape(
                            masked_lm_weights, [-1])
                        masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
                            labels=masked_lm_ids,
                            predictions=masked_lm_predictions,
                            weights=masked_lm_weights)
                        masked_lm_mean_loss = tf.compat.v1.metrics.mean(
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
    # pylint: disable=attribute-defined-outside-init
    '''Top model for pretrain.
    It's MaskLM + Classification(next sentence prediction)
    '''

    def __call__(self, features, hidden_feature, mode, problem_name):
        mask_lm_top = MaskLM(self.params)
        self.params.share_top['next_sentence'] = 'next_sentence'
        mask_lm_top_result = mask_lm_top(
            features, hidden_feature, mode, problem_name)
        with tf.compat.v1.variable_scope('next_sentence', reuse=tf.compat.v1.AUTO_REUSE):
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
    # pylint: disable=attribute-defined-outside-init
    '''Top model for seq2seq problem.
    This is basically a decoder of encoder-decoder framework.
    Here uses transformer decoder architecture with beam search support.
    '''

    def __call__(self, features, hidden_feature, mode, problem_name):
        self.decoder = load_transformer_model(
            self.params.bert_decoder_config,
            self.params.transformer_decoder_model_loading)
        scope_name = self.params.share_top[problem_name]
        encoder_output = hidden_feature['seq']
        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = features['%s_label_ids' % problem_name]
            label_mask = features['{}_mask'.format(problem_name)]
            encoder_mask = features['model_input_mask']

            batch_loss, logits = self.decoder({'input_ids': labels,
                                               'attention_mask': label_mask,
                                               'encoder_hidden_states': encoder_output,
                                               'encoder_attention_mask': encoder_mask,
                                               'labels': labels})

            # loss = self.create_loss(
            #     batch_loss, features['%s_loss_multiplier' % problem_name])
            # # If a batch does not contain input instances from the current problem, the loss multiplier will be empty
            # # and loss will be NaN. Replacing NaN with 0 fixes the problem.
            # loss = tf.compat.v1.where(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
            #     tf.math.is_nan(loss), tf.zeros_like(loss), loss)
            loss = tf.reduce_mean(batch_loss)
            self.loss = loss

            if mode == tf.estimator.ModeKeys.TRAIN:
                return self.loss
            else:
                return self.eval_metric_fn(
                    features, logits, loss, problem_name, features['%s_mask' % problem_name])

        else:
            bos_id = self.params.bos_id
            init_tensor = tf.ones(
                (tf.shape(encoder_output)[0], 1), dtype=tf.int32) * bos_id
            eos_id = self.params.eos_id
            self.pred = tf.identity(self.decoder.generate(
                input_ids=init_tensor,
                max_length=self.params.decode_max_seq_len,
                min_length=2,
                early_stopping=True,
                num_beams=self.params.beam_size,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                use_cache=True
            ),
                name='%s_predict' % scope_name)
            return self.pred


class MultiLabelClassification(TopLayer):
    # pylint: disable=attribute-defined-outside-init
    '''Top model for multi-class classification.
    It's a dense net with body output features as input with following support.

    label_smoothing: Soft label smoothing.
    '''

    def create_batch_loss(self, labels, logits,  num_classes):
        labels = tf.cast(labels, tf.float32)
        batch_label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)

        batch_loss = tf.reduce_sum(input_tensor=batch_label_loss, axis=1)

        if self.params.uncertain_weight_loss:
            batch_loss = self.uncertainty_weighted_loss(batch_loss)
        return batch_loss

    def __call__(self, features, hidden_feature, mode, problem_name, mask=None):
        hidden_feature = hidden_feature['pooled']
        scope_name = self.params.share_top[problem_name]
        if mode == tf.estimator.ModeKeys.TRAIN:
            hidden_feature = tf.nn.dropout(
                hidden_feature,
                rate=1 - (self.params.dropout_keep_prob))

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
            self.loss = tf.compat.v1.where(tf.math.is_nan(self.loss),
                                           tf.zeros_like(self.loss), self.loss)
            return self.loss
        elif mode == tf.estimator.ModeKeys.EVAL:
            labels = features['%s_label_ids' % problem_name]
            batch_loss = self.create_batch_loss(labels, logits, num_classes)
            # multiply with loss multiplier to make some loss as zero
            loss = tf.reduce_mean(input_tensor=batch_loss)
            prob = tf.nn.sigmoid(logits)
            prob = tf.round(prob)
            prob = tf.expand_dims(prob, -1)
            return self.eval_metric_fn(
                features, prob, loss, problem_name)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            prob = tf.nn.sigmoid(logits)
            self.prob = tf.identity(prob, name='%s_predict' % scope_name)
            return self.prob
