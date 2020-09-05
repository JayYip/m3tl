import transformers
from bert_multitask_learning.params import BaseParams
import tensorflow as tf
from typing import Dict, Tuple

from . import modeling
from .utils import load_transformer_model
from .top_utils import gather_indexes


@tf.function
def empty_tensor_handling_loss(labels, logits, loss_fn):
    if tf.equal(tf.size(labels), 0):
        return 0.0
    if tf.equal(tf.size(tf.shape(labels)), 0):
        return 0.0
    if tf.equal(tf.shape(labels)[0], 0):
        return 0.0
    else:
        return tf.reduce_mean(loss_fn(
            labels, logits, from_logits=True))


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

            loss = empty_tensor_handling_loss(
                labels, logits,
                tf.keras.losses.sparse_categorical_crossentropy)
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
        self.metric_fn = tf.keras.metrics.SparseCategoricalAccuracy(
            name='{}_acc'.format(self.problem_name))

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
            loss = empty_tensor_handling_loss(
                labels, logits,
                tf.keras.losses.sparse_categorical_crossentropy)
            self.add_loss(loss)
            acc = self.metric_fn(labels, logits)
            self.add_metric(acc)
        return tf.nn.softmax(
            logits, name='%s_predict' % self.problem_name)


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
        seq_hidden_feature = hidden_features['seq']
        positions = features['masked_lm_positions']

        # gather_indexes will flatten the seq hidden_states, we need to reshape
        # back to 3d tensor
        input_tensor = gather_indexes(seq_hidden_feature, positions)
        shape_tensor = tf.shape(positions)
        shape_list = tf.concat([shape_tensor, [-1]], axis=0)
        input_tensor = tf.reshape(input_tensor, shape=shape_list)
        # set_shape to determin rank
        input_tensor.set_shape(
            [None, None, seq_hidden_feature.shape.as_list()[-1]])
        mlm_logits = self.mlm(input_tensor)

        if mode != tf.estimator.ModeKeys.PREDICT:
            nsp_labels = tf.squeeze(
                features['next_sentence_label_ids'])
            mlm_labels = features['masked_lm_ids']
            mlm_labels.set_shape([None, None])
            # compute loss
            nsp_loss = empty_tensor_handling_loss(
                nsp_labels, nsp_logits,
                tf.keras.losses.sparse_categorical_crossentropy)
            mlm_loss_layer = transformers.modeling_tf_utils.TFMaskedLanguageModelingLoss()
            # mlm_loss = tf.reduce_mean(
            #     mlm_loss_layer.compute_loss(mlm_labels, mlm_logits))

            # add a useless from_logits argument to match the function signature of keras losses.
            def loss_fn_wrapper(labels, logits, from_logits=True):
                return mlm_loss_layer.compute_loss(labels, logits)
            mlm_loss = empty_tensor_handling_loss(
                mlm_labels,
                mlm_logits,
                loss_fn_wrapper
            )
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
            loss = empty_tensor_handling_loss(
                labels, logits, tf.keras.losses.binary_crossentropy)
            self.add_loss(loss)
        return tf.nn.sigmoid(
            logits, name='%s_predict' % self.problem_name)
