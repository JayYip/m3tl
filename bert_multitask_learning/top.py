from build.lib.bert_multitask_learning.top import Seq2Seq
import transformers
from bert_multitask_learning.params import BaseParams
import tensorflow as tf
from typing import Dict, Tuple

from . import modeling
from .utils import load_transformer_model


class SequenceLabel(tf.keras.Model):
    def __init__(self, params: BaseParams, problem_name: str):
        super(SequenceLabel, self).__init__(name=problem_name)
        self.params = params
        self.problem_name = problem_name
        num_classes = self.params.num_classes[self.problem_name]
        self.dense = tf.keras.layers.Dense(num_classes, activation=None)

        self.dropout = tf.keras.layers.Dropout(1-params.dropout_keep_prob)

    def call(self, inputs, mode):
        feature, hidden_feature = inputs
        hidden_feature = hidden_feature['seq']
        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = feature['{}_label_ids'.format(self.problem_name)]
        else:
            labels = None

        hidden_feature = self.dropout(hidden_feature)

        logits = self.dense(hidden_feature)

        if mode != tf.estimator.ModeKeys.PREDICT:
            # inconsistent shape might be introduced to labels
            # so we need to do some padding to make sure that
            # labels has the same sequence length as logits
            pad_len = tf.shape(input=logits)[1] - tf.shape(input=labels)[1]

            # top, bottom, left, right
            pad_tensor = [[0, 0], [0, pad_len]]
            labels = tf.pad(tensor=labels, paddings=pad_tensor)

            batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
            loss = tf.reduce_mean(batch_loss)
            self.add_loss(loss)
        return tf.nn.softmax(
            logits, name='%s_predict' % self.problem_name)


class Classification(tf.keras.layers.Layer):
    def __init__(self, params: BaseParams, problem_name: str) -> None:
        super(Classification, self).__init__(name=problem_name)
        self.params = params
        self.problem_name = problem_name
        num_classes = self.params.num_classes[self.problem_name]
        self.dense = tf.keras.layers.Dense(num_classes, activation=None)

        self.dropout = tf.keras.layers.Dropout(1-params.dropout_keep_prob)

    def call(self, inputs, mode):
        feature, hidden_feature = inputs
        hidden_feature = hidden_feature['pooled']
        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = feature['{}_label_ids'.format(self.problem_name)]
        else:
            labels = None
        hidden_feature = self.dropout(hidden_feature)
        logits = self.dense(hidden_feature)

        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = tf.squeeze(labels)
            batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
            loss = tf.reduce_mean(batch_loss)
            self.add_loss(loss)
        return tf.nn.softmax(
            logits, name='%s_predict' % self.problem_name)


# class MaskLM(TopLayer):
#     # pylint: disable=attribute-defined-outside-init
#     '''Top model for mask language model.
#     It's a dense net with body output features as input.
#     Major logic is from original bert code
#     '''

#     def __call__(self, features, hidden_feature, mode, problem_name):
#         """Get loss and log probs for the masked LM.

#         DO NOT CHANGE THE VARAIBLE SCOPE.
#         """
#         seq_hidden_feature = hidden_feature['seq']
#         positions = features['masked_lm_positions']
#         input_tensor = gather_indexes(seq_hidden_feature, positions)
#         output_weights = hidden_feature['embed_table']
#         label_ids = features['masked_lm_ids']
#         label_weights = features['masked_lm_weights']

#         with tf.compat.v1.variable_scope("cls/predictions"):
#             # We apply one more non-linear transformation before the output layer.
#             # This matrix is not used after pre-training.
#             with tf.compat.v1.variable_scope("transform"):
#                 input_tensor = tf.compat.v1.layers.dense(
#                     input_tensor,
#                     units=self.params.mask_lm_hidden_size,
#                     activation=modeling.get_activation(
#                         self.params.mask_lm_hidden_act),
#                     kernel_initializer=modeling.create_initializer(
#                         self.params.mask_lm_initializer_range))
#                 input_tensor = modeling.layer_norm(input_tensor)

#             # The output weights are the same as the input embeddings, but there is
#             # an output-only bias for each token.
#             output_bias = tf.compat.v1.get_variable(
#                 "output_bias",
#                 shape=[self.params.vocab_size],
#                 initializer=tf.compat.v1.zeros_initializer())

#             logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
#             logits = tf.nn.bias_add(logits, output_bias)
#             self.logits = logits
#             log_probs = tf.nn.log_softmax(logits, axis=-1)

#             if mode == tf.estimator.ModeKeys.PREDICT:
#                 self.prob = log_probs
#                 return self.prob

#             else:

#                 label_ids = tf.reshape(label_ids, [-1])
#                 label_weights = tf.reshape(label_weights, [-1])

#                 one_hot_labels = tf.one_hot(
#                     label_ids, depth=self.params.vocab_size, dtype=tf.float32)

#                 # The `positions` tensor might be zero-padded (if the sequence is too
#                 # short to have the maximum number of predictions). The `label_weights`
#                 # tensor has a value of 1.0 for every real prediction and 0.0 for the
#                 # padding predictions.
#                 per_example_loss = - \
#                     tf.reduce_sum(input_tensor=log_probs *
#                                   one_hot_labels, axis=[-1])
#                 label_weights = tf.cast(label_weights, tf.float32)
#                 numerator = tf.reduce_sum(
#                     input_tensor=label_weights * per_example_loss)
#                 denominator = tf.reduce_sum(input_tensor=label_weights) + 1e-5
#                 loss = numerator / denominator

#                 if mode == tf.estimator.ModeKeys.TRAIN:
#                     self.loss = loss
#                     return self.loss

#                 else:
#                     def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
#                                   masked_lm_weights):
#                         """Computes the loss and accuracy of the model."""
#                         masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
#                                                          [-1, masked_lm_log_probs.shape[-1]])
#                         masked_lm_predictions = tf.argmax(
#                             input=masked_lm_log_probs, axis=-1, output_type=tf.int32)
#                         masked_lm_example_loss = tf.reshape(
#                             masked_lm_example_loss, [-1])
#                         masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
#                         masked_lm_weights = tf.reshape(
#                             masked_lm_weights, [-1])
#                         masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
#                             labels=masked_lm_ids,
#                             predictions=masked_lm_predictions,
#                             weights=masked_lm_weights)
#                         masked_lm_mean_loss = tf.compat.v1.metrics.mean(
#                             values=masked_lm_example_loss, weights=masked_lm_weights)

#                         return {
#                             "masked_lm_accuracy": masked_lm_accuracy,
#                             "masked_lm_loss": masked_lm_mean_loss,
#                         }
#                     eval_metrics = (metric_fn(
#                         per_example_loss, log_probs, label_ids,
#                         label_weights), loss)

#                     self.eval_metrics = eval_metrics
#                     return self.eval_metrics


class PreTrain(tf.keras.Model):
    def __init__(self, params: BaseParams, problem_name: str, input_embeddings: tf.Tensor):
        super(PreTrain, self).__init__(name=problem_name)
        self.params = params
        self.nsp = transformers.modeling_tf_bert.TFBertNSPHead(
            self.params.bert_config)

        # TODO: add mlm back to pretrain
        self.mlm = transformers.modeling_tf_bert.TFBertMLMHead(
            self.params.bert_config, input_embeddings=input_embeddings)

    def call(self,
             inputs: Tuple[Dict[str, Dict[str, tf.Tensor]], Dict[str, Dict[str, tf.Tensor]]],
             mode: str) -> Tuple[tf.Tensor, tf.Tensor]:
        features, hidden_features = inputs

        # compute logits
        nsp_logits = self.nsp(hidden_features['pooled'])

        # masking is done inside the model
        mlm_logits = self.mlm(hidden_features['seq'])

        if mode != tf.estimator.ModeKeys.PREDICT:
            nsp_labels = features['next_sentence_label_ids']
            mlm_labels = features['masked_lm_ids']
            # compute loss
            nsp_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                nsp_labels, nsp_logits, from_logits=True))
            # mlm_loss = tf.reduce_mean(transformers.modeling_tf_utils.TFMaskedLanguageModelingLoss(
            #     labels=mlm_labels, logits=mlm_logits))
            mlm_loss = 0
            loss = nsp_loss + mlm_loss
            self.add_loss(loss)

        return (tf.sigmoid(nsp_logits), tf.nn.softmax(mlm_logits))


class Seq2Seq(tf.keras.Model):
    def __init__(self, params: BaseParams, problem_name: str):
        super(Seq2Seq, self).__init__(name=problem_name)
        self.params = params
        self.problem_name = problem_name
        self.decoder = load_transformer_model(
            self.params.transformer_decoder_model_name,
            self.params.transformer_decoder_model_loading)

    def call(self,
             inputs: Tuple[Dict[str, Dict[str, tf.Tensor]], Dict[str, Dict[str, tf.Tensor]]],
             mode: str):
        features, hidden_features = inputs
        encoder_output = hidden_features['seq']
        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = features['%s_label_ids' % self.problem_name]
            label_mask = features['{}_mask'.format(self.problem_name)]
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
            self.add_loss(loss)
            return tf.nn.softmax(logits)

        else:
            bos_id = self.params.bos_id
            init_tensor = tf.ones(
                (tf.shape(encoder_output)[0], 1), dtype=tf.int32) * bos_id
            eos_id = self.params.eos_id
            pred = self.decoder.generate(
                input_ids=init_tensor,
                max_length=self.params.decode_max_seq_len,
                min_length=2,
                early_stopping=True,
                num_beams=self.params.beam_size,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                use_cache=True
            )
            return pred


# class Seq2Seq(TopLayer):
#     # pylint: disable=attribute-defined-outside-init
#     '''Top model for seq2seq problem.
#     This is basically a decoder of encoder-decoder framework.
#     Here uses transformer decoder architecture with beam search support.
#     '''

#     def __call__(self, features, hidden_feature, mode, problem_name):
#         self.decoder = load_transformer_model(
#             self.params.transformer_decoder_model_name,
#             self.params.transformer_decoder_model_loading)
#         scope_name = self.params.share_top[problem_name]
#         encoder_output = hidden_feature['seq']
#         if mode != tf.estimator.ModeKeys.PREDICT:
#             labels = features['%s_label_ids' % problem_name]
#             label_mask = features['{}_mask'.format(problem_name)]
#             encoder_mask = features['model_input_mask']

#             batch_loss, logits = self.decoder({'input_ids': labels,
#                                                'attention_mask': label_mask,
#                                                'encoder_hidden_states': encoder_output,
#                                                'encoder_attention_mask': encoder_mask,
#                                                'labels': labels})

#             # loss = self.create_loss(
#             #     batch_loss, features['%s_loss_multiplier' % problem_name])
#             # # If a batch does not contain input instances from the current problem, the loss multiplier will be empty
#             # # and loss will be NaN. Replacing NaN with 0 fixes the problem.
#             # loss = tf.compat.v1.where(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
#             #     tf.math.is_nan(loss), tf.zeros_like(loss), loss)
#             loss = tf.reduce_mean(batch_loss)
#             self.loss = loss

#             if mode == tf.estimator.ModeKeys.TRAIN:
#                 return self.loss
#             else:
#                 return self.eval_metric_fn(
#                     features, logits, loss, problem_name, features['%s_mask' % problem_name])

#         else:
#             bos_id = self.params.bos_id
#             init_tensor = tf.ones(
#                 (tf.shape(encoder_output)[0], 1), dtype=tf.int32) * bos_id
#             eos_id = self.params.eos_id
#             self.pred = tf.identity(self.decoder.generate(
#                 input_ids=init_tensor,
#                 max_length=self.params.decode_max_seq_len,
#                 min_length=2,
#                 early_stopping=True,
#                 num_beams=self.params.beam_size,
#                 bos_token_id=bos_id,
#                 eos_token_id=eos_id,
#                 use_cache=True
#             ),
#                 name='%s_predict' % scope_name)
#             return self.pred


class MultiLabelClassification(tf.keras.layers.Layer):
    def __init__(self, params: BaseParams, problem_name: str) -> None:
        super(MultiLabelClassification, self).__init__(name=problem_name)
        self.params = params
        self.problem_name = problem_name
        self.dense = tf.keras.layers.Dense(
            self.params.num_classes[problem_name])
        self.dropout = tf.keras.layers.Dropout(
            1-self.params.dropout_keep_prob
        )

    def call(self, inputs, mode):
        feature, hidden_feature = inputs
        hidden_feature = hidden_feature['pooled']
        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = feature['{}_label_ids'.format(self.problem_name)]
        else:
            labels = None
        hidden_feature = self.dropout(hidden_feature)
        logits = self.dense(hidden_feature)

        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = tf.squeeze(labels)
            labels = tf.cast(labels, tf.float32)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits))
            self.add_loss(loss)
        return tf.nn.sigmoid(
            logits, name='%s_predict' % self.problem_name)
