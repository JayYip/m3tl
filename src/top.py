import tensorflow as tf

from tensor2tensor.utils import metrics

from .t2t_utils import get_t2t_metric_op


def cls(model, features, hidden_feature, mode, problem_name):
    hidden_feature = hidden_feature['pooled']
    if mode == tf.estimator.ModeKeys.TRAIN:
        hidden_feature = tf.nn.dropout(
            hidden_feature,
            keep_prob=model.config.dropout_keep_prob)

    num_classes = model.config.num_classes[problem_name]
    output_layer = tf.layers.Dense(
        num_classes, activation=None,
        kernel_initializer=tf.orthogonal_initializer()
    )
    logits = output_layer(hidden_feature)
    labels = features['%s_label_ids' % problem_name]
    if mode == tf.estimator.ModeKeys.TRAIN:
        batch_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        loss = tf.reduce_mean(batch_loss)

        tf.summary.scalar('%s_loss' % problem_name, loss)
        return loss
    elif mode == tf.estimator.ModeKeys.EVAL:
        batch_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        loss = tf.reduce_mean(batch_loss)

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
        return eval_metrics
    else:
        prob = tf.nn.softmax(logits)
        return prob
    return None


def seq_tag(model, features, hidden_feature, mode, problem_name):
    hidden_feature = hidden_feature['seq']
    if mode == tf.estimator.ModeKeys.TRAIN:
        hidden_feature = tf.nn.dropout(
            hidden_feature,
            keep_prob=model.config.dropout_keep_prob)

    num_classes = model.config.num_classes[problem_name]

    output_layer = tf.layers.Dense(
        num_classes, activation=None,
        kernel_initializer=tf.orthogonal_initializer()
    )
    logits = output_layer(hidden_feature)

    # CRF transition param
    crf_transition_param = tf.get_variable(
        'crf_transition', shape=[num_classes, num_classes])

    # sequence_weight = tf.cast(features["input_mask"], tf.float32)
    seq_length = tf.reduce_sum(features["input_mask"], axis=-1)

    if mode == tf.estimator.ModeKeys.TRAIN:
        seq_labels = features['%s_label_ids' % problem_name]
        with tf.variable_scope('CRF'):
            log_likelihood, _ =\
                tf.contrib.crf.crf_log_likelihood(
                    logits, seq_labels, seq_length,
                    transition_params=crf_transition_param)
        # seq_loss = tf.contrib.seq2seq.sequence_loss(
        #     logits, seq_labels, weights=sequence_weight)
        seq_loss = tf.reduce_mean(-log_likelihood)
        tf.summary.scalar('%s_loss' % problem_name, seq_loss)
        return seq_loss

    elif mode == tf.estimator.ModeKeys.EVAL:
        seq_labels = features['%s_label_ids' % problem_name]
        with tf.variable_scope('CRF'):

            log_likelihood, _ =\
                tf.contrib.crf.crf_log_likelihood(
                    logits, seq_labels, seq_length,
                    transition_params=crf_transition_param)

        # calculate  eval loss
        # seq_loss = tf.contrib.seq2seq.sequence_loss(
        #     logits, seq_labels, weights=sequence_weight)
        seq_loss = tf.reduce_mean(-log_likelihood)

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
        return eval_metrics
    else:
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
            logits, crf_transition_param, seq_length)
        # prob = tf.nn.softmax(logits)
        return viterbi_sequence
