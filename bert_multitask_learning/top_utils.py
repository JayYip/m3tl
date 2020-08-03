
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import autograph


from . import modeling


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

    def eval_metric_fn(self, features, logits, loss, problem, weights=None, pad_labels_to_logits=True):
        label_ids = features['%s_label_ids' % problem]

        if pad_labels_to_logits:
            # inconsistent shape might be introduced to labels
            # so we need to do some padding to make sure that
            # seq_labels has the same sequence length as logits
            pad_len = tf.shape(logits)[1] - tf.shape(label_ids)[1]

            # top, bottom, left, right
            pad_tensor = [[0, 0], [0, pad_len]]
            label_ids = tf.pad(label_ids, paddings=pad_tensor)

        def metric_fn(label_ids, logits):
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

            accuracy = tf.metrics.accuracy(
                label_ids, predictions, weights=weights)

            return {
                "Accuracy": accuracy
            }
        eval_metrics = (metric_fn(label_ids, logits), loss)
        self.eval_metrics = eval_metrics
        return self.eval_metrics

    def uncertainty_weighted_loss(self, loss):
        """Uncertainty to weight losses

        Ref: https://arxiv.org/abs/1705.07115

        Arguments:
            loss {Tensor} -- Batch loss tensor

        Returns:
            Tensor -- Weighted batch loss tensor
        """
        log_var = tf.get_variable(
            shape=(), name='log_var', initializer=tf.zeros_initializer())
        precision = tf.exp(-log_var)
        new_loss = precision*loss + log_var
        return new_loss

    def create_loss(self, batch_loss, loss_multiplier):
        '''This helper function is used to multiply batch loss
        with loss multiplier

        Arguments:
            batch_loss {} -- batch loss
            loss_multiplier {} -- loss multiplier
        '''
        loss_multiplier = tf.cast(
            loss_multiplier, tf.float32)
        # multiply with loss multiplier to make some loss as zero
        loss = tf.reduce_mean(batch_loss * loss_multiplier)
        # if batch_loss is empty, loss will be nan, replace with zero
        loss = tf.where(tf.is_nan(loss),
                        tf.zeros_like(loss), loss)

        tf.summary.scalar('loss', loss)
        return loss

    def make_hidden_model(self, features, hidden_feature, mode, is_seq=False):

        if self.params.hidden_gru and is_seq:
            with tf.variable_scope('hidden'):
                new_hidden_feature = make_cudnngru(
                    hidden_feature,
                    int(self.params.bert_config.hidden_size / 2),
                    self.params,
                    mode)

                new_hidden_feature.set_shape(
                    [None, self.params.max_seq_len, self.params.bert_config.hidden_size])

                self.hidden_model_logit = new_hidden_feature

                return new_hidden_feature
        elif self.params.hidden_dense:
            with tf.variable_scope('hidden'):
                hidden_feature = dense_layer(
                    self.params.bert_config.hidden_size,
                    hidden_feature, mode,
                    self.params.dropout_keep_prob,
                    tf.nn.relu)
                self.hidden_model_logit = hidden_feature
            return hidden_feature
        else:
            return hidden_feature

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
    flat_offsets = tf.cast(flat_offsets, tf.int64)
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


def create_seq_smooth_label(params, labels, num_classes):
    # since crf dose not take the smoothed label, consider the
    # 'hard' smoothing. That is, sample a tag based on smooth factor
    if params.label_smoothing > 0:

        true_labels = tf.stack(
            [labels]*int(num_classes/params.label_smoothing), axis=-1)
        single_label_set = tf.stack([tf.range(
            num_classes)]*params.max_seq_len, axis=0)
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


def dense_layer(hidden_size, hidden_feature, mode, dropout_keep_prob, activation):
    # simple wrapper of dense layer
    output_layer = tf.layers.Dense(
        hidden_size, activation=activation,
        kernel_initializer=tf.orthogonal_initializer()
    )
    hidden_logits = output_layer(hidden_feature)
    if mode == tf.estimator.ModeKeys.TRAIN:
        hidden_logits = tf.nn.dropout(
            hidden_logits,
            keep_prob=dropout_keep_prob)
    return hidden_logits
