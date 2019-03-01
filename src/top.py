from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import autograph
from copy import copy


from tensor2tensor.utils import metrics
from tensor2tensor.utils import beam_search

from .t2t_utils import get_t2t_metric_op
from .transformer_decoder import TransformerDecoder

from .bert import modeling


class TopLayer():
    def __init__(self, params):
        self.params = params

    def get_train(self):
        return self.loss

    def get_eval(self):
        return self.eval_metrics

    def get_predict(self):
        return self.prob

    def get_logit(self):
        return self.logits

    def __call__(self, features, hidden_feature, mode, problem_name):
        raise NotImplementedError


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


@autograph.convert()
def _make_cudnngru(
        hidden_feature,
        hidden_size,
        output_hidden_size,
        merge_mode='concat',
        dropout_keep_prob=1.0):

    if tf.shape(hidden_feature)[0] == 0:
        rnn_output = tf.zeros(
            [0, tf.shape(hidden_feature)[1], output_hidden_size])
    else:

        rnn_layer = keras.layers.CuDNNGRU(
            units=hidden_size,
            return_sequences=True,
            return_state=False)
        rnn_output = keras.layers.Bidirectional(
            layer=rnn_layer,
            merge_mode=merge_mode)(hidden_feature)
        rnn_output = tf.nn.dropout(rnn_output, keep_prob=dropout_keep_prob)
        rnn_output = keras.layers.ReLU()(rnn_output)

    return rnn_output


def make_cudnngru(
        hidden_feature,
        hidden_size,
        params,
        mode,
        res_connection=True,
        merge_mode='concat'):

    if merge_mode == 'concat':
        output_hidden_size = 2*hidden_size
    else:
        output_hidden_size = hidden_size

    rnn_output = _make_cudnngru(
        hidden_feature, hidden_size, output_hidden_size, merge_mode, params.dropout_keep_prob)

    rnn_output.set_shape(
        [None, params.max_seq_len, output_hidden_size])

    if res_connection:
        hidden_feature_size = hidden_feature.get_shape().as_list()[-1]
        if hidden_size != tf.shape(hidden_feature)[-1]:
            with tf.variable_scope('hidden_gru_projection'):
                rnn_output = tf.layers.Dense(
                    hidden_feature_size, activation=None,
                    kernel_initializer=tf.orthogonal_initializer())(rnn_output)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    rnn_output = tf.nn.dropout(
                        rnn_output, params.dropout_keep_prob)

        rnn_output = rnn_output + hidden_feature

    if mode == tf.estimator.ModeKeys.TRAIN:

        # res connection and layer norm
        hidden_feature = modeling.layer_norm_and_dropout(
            rnn_output,
            1 - params.dropout_keep_prob)
    else:
        hidden_feature = modeling.layer_norm(
            rnn_output
        )
    return hidden_feature


class SequenceLabel(TopLayer):
    '''Top model for sequence labeling.
    It's a dense net with body output features as input with following support.

    crf: Conditional Random Field. Take logits(output of dense layer) as input
    hidden_gru: Take body features as input and apply rnn on it.
    label_smoothing: Hard label smoothing. Random replace label by some prob.
    '''

    def create_smooth_label(self, labels, num_classes):
        # since crf dose not take the smoothed label, consider the
        # 'hard' smoothing. That is, sample a tag based on smooth factor
        if self.params.label_smoothing > 0:

            true_labels = tf.stack(
                [labels]*int(num_classes/self.params.label_smoothing), axis=-1)
            single_label_set = tf.stack([tf.range(
                num_classes)]*self.params.max_seq_len, axis=0)
            batch_size_this_turn = tf.shape(true_labels)[0]
            label_set = tf.broadcast_to(
                input=single_label_set, shape=[
                    batch_size_this_turn,
                    single_label_set.shape.as_list()[
                        0],
                    single_label_set.shape.as_list()[1]])
            sample_set = tf.concat([true_labels, label_set], axis=-1)

            dims = tf.shape(sample_set)
            sample_set = tf.reshape(sample_set, shape=[-1, dims[-1]])

            samples_index = tf.random_uniform(
                shape=[tf.shape(sample_set)[0], 1], minval=0, maxval=tf.shape(sample_set)[1], dtype=tf.int32)
            flat_offsets = tf.reshape(
                tf.range(0, tf.shape(sample_set)[0], dtype=tf.int32) * tf.shape(sample_set)[1], [-1, 1])
            flat_index = tf.reshape(samples_index+flat_offsets, [-1])
            sampled_label = tf.gather(
                tf.reshape(sample_set, [-1]), flat_index)
            sampled_label = tf.reshape(sampled_label, dims[:-1])
        else:
            sampled_label = labels
        return sampled_label

    def make_batch_loss(self, logits, seq_labels, seq_length, crf_transition_param):
        if self.params.crf:
            with tf.variable_scope('CRF'):
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    logits, seq_labels, seq_length,
                    transition_params=crf_transition_param)
                batch_loss = -log_likelihood
        else:
            batch_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=seq_labels), axis=1)
        return batch_loss

    def make_hidden_model(self, features, hidden_feature, mode):
        if self.params.hidden_gru:
            with tf.variable_scope('hidden'):
                new_hidden_feature = make_cudnngru(
                    hidden_feature,
                    int(self.params.bert_config.hidden_size / 2),
                    self.params,
                    mode)

                new_hidden_feature.set_shape(
                    [None, self.params.max_seq_len, self.params.bert_config.hidden_size])

                return new_hidden_feature
        else:
            return hidden_feature

    def __call__(self, features, hidden_feature, mode, problem_name, mask=None):
        hidden_feature = hidden_feature['seq']
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
            features, hidden_feature, mode)

        output_layer = tf.layers.Dense(
            num_classes, activation=None,
            kernel_initializer=tf.orthogonal_initializer()
        )
        logits = output_layer(hidden_feature)
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
            seq_labels = self.create_smooth_label(seq_labels, num_classes)
            batch_loss = self.make_batch_loss(
                logits, seq_labels, seq_length, crf_transition_param)
            loss_multiplier = tf.cast(
                features['%s_loss_multiplier' % problem_name], tf.float32)
            # multiply with loss multiplier to make some loss as zero
            seq_loss = tf.reduce_mean(batch_loss * loss_multiplier)
            tf.summary.scalar('%s_loss' % problem_name, seq_loss)
            self.loss = seq_loss
            return self.loss

        elif mode == tf.estimator.ModeKeys.EVAL:
            seq_labels = features['%s_label_ids' % problem_name]
            batch_loss = self.make_batch_loss(
                logits, seq_labels, seq_length, crf_transition_param)

            seq_loss = tf.reduce_mean(batch_loss)

            def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                prob = tf.nn.softmax(logits)

                accuracy = tf.metrics.accuracy(
                    label_ids, predictions, weights=features['input_mask'])
                acc_per_seq = get_t2t_metric_op(metrics.METRICS_FNS[
                    metrics.Metrics.ACC_PER_SEQ],
                    prob, features, label_ids)
                one_hot_labels = tf.one_hot(
                    label_ids, depth=num_classes)
                f1_score = tf.contrib.metrics.f1_score(
                    one_hot_labels, prob, weights=features['input_mask'])

                return {
                    "Accuracy": accuracy,
                    'Accuracy Per Sequence': acc_per_seq,
                    'F1 Score': f1_score
                }

            eval_metrics = (metric_fn(seq_labels, logits), seq_loss)
            self.eval_metrics = eval_metrics
            return self.eval_metrics
        elif mode == tf.estimator.ModeKeys.PREDICT:
            if self.params.crf:
                viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
                    logits, crf_transition_param, seq_length)
                self.prob = tf.identity(
                    viterbi_sequence, name='%s_predict' % problem_name)
            else:
                self.prob = tf.nn.softmax(
                    logits, name='%s_predict' % problem_name)

            return self.prob


class Classification(TopLayer):
    '''Top model for classification.
    It's a dense net with body output features as input with following support.

    label_smoothing: Soft label smoothing.
    '''

    def create_loss(self, labels, logits,  num_classes):
        if self.params.label_smoothing > 0:
            one_hot_labels = tf.one_hot(labels, depth=num_classes)
            return tf.losses.softmax_cross_entropy(
                one_hot_labels, logits,
                label_smoothing=self.params.label_smoothing)
        else:
            return tf.losses.sparse_softmax_cross_entropy(labels, logits)

    def __call__(self, features, hidden_feature, mode, problem_name, mask=None):
        hidden_feature = hidden_feature['pooled']
        if mode == tf.estimator.ModeKeys.TRAIN:
            hidden_feature = tf.nn.dropout(
                hidden_feature,
                keep_prob=self.params.dropout_keep_prob)

        if mask is None:
            num_classes = self.params.num_classes[problem_name]
        else:
            num_classes = mask.shape[0]

        output_layer = tf.layers.Dense(
            num_classes, activation=None,
            kernel_initializer=tf.orthogonal_initializer()
        )
        logits = output_layer(hidden_feature)
        self.logits = logits
        if mask is not None:
            logits = logits*mask
        if mode == tf.estimator.ModeKeys.TRAIN:
            labels = features['%s_label_ids' % problem_name]
            batch_loss = self.create_loss(labels, logits, num_classes)
            loss_multiplier = tf.cast(
                features['%s_loss_multiplier' % problem_name], tf.float32)
            # multiply with loss multiplier to make some loss as zero
            loss = tf.reduce_mean(batch_loss*loss_multiplier)

            tf.summary.scalar('%s_loss' % problem_name, loss)
            self.loss = loss
            return self.loss
        elif mode == tf.estimator.ModeKeys.EVAL:
            labels = features['%s_label_ids' % problem_name]
            batch_loss = self.create_loss(labels, logits, num_classes)
            loss_multiplier = tf.cast(
                features['%s_loss_multiplier' % problem_name], tf.float32)
            # multiply with loss multiplier to make some loss as zero
            loss = tf.reduce_mean(batch_loss*loss_multiplier)

            def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                prob = tf.nn.softmax(logits)
                accuracy = tf.metrics.accuracy(
                    label_ids, predictions)
                one_hot_labels = tf.one_hot(
                    label_ids, depth=num_classes)
                f1_score = tf.contrib.metrics.f1_score(
                    one_hot_labels, prob)

                return {
                    "Accuracy": accuracy,
                    'F1 Score': f1_score
                }
            eval_metrics = (metric_fn(labels, logits), loss)
            self.eval_metrics = eval_metrics
            return self.eval_metrics
        elif mode == tf.estimator.ModeKeys.PREDICT:
            prob = tf.nn.softmax(logits)
            self.prob = tf.identity(prob, name='%s_predict' % problem_name)
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
        cls = Classification(self.params)
        mask_lm_top_result = mask_lm_top(
            features, hidden_feature, mode, problem_name)
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

        for problem_dict in self.params.label_transfer_problem:
            for problem in problem_dict:
                if problem in self.params.share_top:
                    top_name = self.params.share_top[problem]

                else:
                    top_name = problem

                top_scope_name = '%s_top' % top_name

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
            for problem_dict in self.params.label_transfer_problem:
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

    def create_loss(self, labels, logits, features, problem_name):
        batch_loss = tf.losses.sparse_softmax_cross_entropy(
            labels, logits)
        loss_multiplier = tf.cast(
            features['%s_loss_multiplier' % problem_name], tf.float32)
        # multiply with loss multiplier to make some loss as zero
        loss = tf.reduce_mean(batch_loss*loss_multiplier)

        return loss

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

        decode_ids, _ = beam_search.beam_search(
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

        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = features['%s_label_ids' % problem_name]

            logits = self.decoder.train_eval(
                features, hidden_feature, mode, problem_name)

            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                shift_labels = tf.pad(
                    labels, [[0, 0], [0, 1]])[:, 1:]

            loss = self.create_loss(
                shift_labels, logits, features, problem_name)

            tf.summary.scalar('%s_loss' % problem_name, loss)
            self.loss = loss

            if mode == tf.estimator.ModeKeys.TRAIN:
                return self.loss

            def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                prob = tf.nn.softmax(logits)

                accuracy = tf.metrics.accuracy(
                    label_ids, predictions, weights=features['%s_mask' % problem_name])
                acc_per_seq = get_t2t_metric_op(metrics.METRICS_FNS[
                    metrics.Metrics.ACC_PER_SEQ],
                    prob, features, label_ids)

                blue_prob = tf.expand_dims(prob, axis=-2)
                blue_prob = tf.expand_dims(blue_prob, axis=-2)

                blue_label_ids = tf.expand_dims(label_ids, axis=-1)
                blue_label_ids = tf.expand_dims(blue_label_ids, axis=-1)
                bleu_score = get_t2t_metric_op(metrics.METRICS_FNS[
                    metrics.Metrics.APPROX_BLEU],
                    blue_prob, features, blue_label_ids)

                return {
                    "Accuracy": accuracy,
                    'Accuracy Per Sequence': acc_per_seq,
                    'Approximate BLEU': bleu_score
                }

            eval_metrics = (metric_fn(labels, logits), loss)
            self.eval_metrics = eval_metrics
            return self.eval_metrics

        else:
            self.pred = tf.identity(self.beam_search_decode(
                features, hidden_feature, mode, problem_name),
                name='%s_predict' % problem_name)
            return self.pred


class MutualPrediction(TopLayer):
    '''Top model for mutual prediction, specific for multitask

    In training, each problem will have three logits:
    1. standard logits
    2. mutual prediction logits, use logits from OTHER problem to predict, for example, 
        a|b|c, for problem a, it's mutual prediction logits is [standard_logits(b), standard_logits(c)]
    3. final logits, use [standard_logits(a), mutual_logits(a)] to do final prediction

    Losses created by above three logits will be added together and optimized.
    '''

    def make_top_layer(self, top_scope_name, top_name, features, hidden_feature, mode, problem):

        with tf.variable_scope(top_scope_name, reuse=tf.AUTO_REUSE):
            if self.params.problem_type[problem] == 'seq_tag':
                seq_tag = SequenceLabel(self.params)
                loss = seq_tag(
                    features,
                    hidden_feature,
                    mode,
                    problem)
                res = seq_tag

            elif self.params.problem_type[problem] == 'cls':
                cls = Classification(self.params)

                loss = cls(
                    features,
                    hidden_feature,
                    mode,
                    problem)
                res = cls
            else:
                return None

            if mode == tf.estimator.ModeKeys.TRAIN:
                tf.summary.scalar(top_scope_name, loss)
            return res

    def __call__(self, features, hidden_feature, mode, problem_name):
        crt_flag = self.params.crf
        self.params.crf = False
        # apply standard top to problems
        single_problem_top = {}
        for problem_dict in self.params.label_transfer_problem:
            for problem in problem_dict:
                if problem in self.params.share_top:
                    top_name = self.params.share_top[problem]

                else:
                    top_name = problem

                top_scope_name = '%s_top' % top_name

                single_problem_top[problem] = self.make_top_layer(
                    top_scope_name,
                    top_name,
                    features,
                    hidden_feature,
                    mode,
                    problem)

        # Mutual prediction top
        mutual_top = {}
        for problem in single_problem_top:
            problem_type = self.params.problem_type[problem]
            if problem in self.params.share_top:
                top_name = self.params.share_top[problem]

            else:
                top_name = problem

            top_scope_name = '%s_mutual' % top_name
            mutual_hidden_feature = defaultdict(list)

            # append logits of same problem type other than
            # the main problem to mutual_hidden_feature
            for side_problem in single_problem_top:
                if side_problem in self.params.share_top:
                    side_top_name = self.params.share_top[side_problem]

                else:
                    side_top_name = side_problem

                if side_top_name != top_name and \
                        self.params.problem_type[side_problem] == problem_type:
                    if problem_type == 'seq_tag':
                        mutual_hidden_feature['seq'].append(
                            single_problem_top[side_top_name].get_logit())
                    elif problem_type == 'cls':
                        mutual_hidden_feature['pooled'].append(
                            single_problem_top[side_top_name].get_logit())
            if problem_type == 'seq_tag':
                mutual_hidden_feature['seq'] = tf.concat(
                    mutual_hidden_feature['seq'], axis=-1)
            elif problem_type == 'cls':
                mutual_hidden_feature['pooled'] = tf.concat(
                    mutual_hidden_feature['pooled'], axis=-1)
            mutual_top[problem] = self.make_top_layer(
                top_scope_name,
                top_name,
                features,
                mutual_hidden_feature,
                mode,
                problem)

        self.params.crf = crt_flag
        # Final prediction
        final_return_dict = {}
        for problem in single_problem_top:
            problem_type = self.params.problem_type[problem]
            if problem_type in ['seq2seq_tag', 'seq2seq_text']:
                final_top_layer = Seq2Seq(self.params)
                loss = final_top_layer(
                    features,
                    hidden_feature,
                    mode,
                    problem)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    tf.summary.scalar(top_scope_name, loss)

            else:
                final_hidden_feature = tf.concat([
                    single_problem_top[problem].get_logit(), mutual_top[problem].get_logit()],
                    axis=-1)
                final_hidden_feature_dict = {
                    'seq': final_hidden_feature, 'pooled': final_hidden_feature}
                if problem in self.params.share_top:
                    top_name = self.params.share_top[problem]

                else:
                    top_name = problem

                top_scope_name = '%s_final_top' % top_name

                final_top_layer = self.make_top_layer(
                    top_scope_name,
                    top_name,
                    features,
                    final_hidden_feature_dict,
                    mode,
                    problem)

            # if is train, add loss of single problem top and mutual top
            if mode == tf.estimator.ModeKeys.TRAIN:
                final_return_dict[problem] = final_top_layer.get_train()
                if problem_type not in ['seq2seq_tag', 'seq2seq_text']:
                    final_return_dict[problem] += single_problem_top[problem].get_train() + \
                        mutual_top[problem].get_train()
            elif mode == tf.estimator.ModeKeys.EVAL:
                final_return_dict[problem] = final_top_layer.get_eval()
            else:
                final_return_dict[problem] = final_top_layer.get_predict()

        return final_return_dict


class GridTransformer(TopLayer):

    def __call__(self, features, hidden_feature, mode, problem_name):
        key_hidden_feature = hidden_feature['all']

        query_hidden_feature = hidden_feature['seq']
        hidden_size = self.params.bert_config.hidden_size

        # transform hidden feature to batch_size, max_seq*num_layers, hidden_size
        key_hidden_feature = tf.reshape(
            key_hidden_feature,
            [-1, self.params.bert_config.num_hidden_layers*self.params.max_seq_len, hidden_size])

        # dense transformation to same shape
        output_layer = tf.layers.Dense(
            hidden_size, activation=tf.nn.relu,
            kernel_initializer=tf.orthogonal_initializer()
        )
        hidden_logits = output_layer(query_hidden_feature)

        grid_transformer_params = copy(self.params)
        grid_transformer_params.decoder_num_hidden_layers = 1
        self.decoder = TransformerDecoder(grid_transformer_params)

        encoder_output = key_hidden_feature
        decoder_inputs = hidden_logits
        input_mask = features['input_mask']
        self_attention_mask = input_mask
        enc_dec_attention_mask = tf.concat(
            [input_mask]*self.params.bert_config.num_hidden_layers, axis=1)

        decode_output = self.decoder.decode(
            decoder_inputs=decoder_inputs,
            encoder_output=encoder_output,
            input_mask=input_mask,
            decoder_self_attention_mask=self_attention_mask,
            cache=None,
            num_classes=None,
            do_return_all_layers=False,
            enc_dec_attention_mask=enc_dec_attention_mask
        )

        return decode_output
