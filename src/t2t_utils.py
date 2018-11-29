import tensorflow as tf
import inspect

from tensor2tensor.layers import common_layers


def reduce_dimensions(predictions, labels):
    """Reduce dimensions for high-dimensional predictions and labels."""
    # We will treat first dimensions as batch. One example are video frames.
    if len(predictions.get_shape()) > 5:
        predictions_shape = common_layers.shape_list(predictions)
        predictions = tf.reshape(
            predictions, [predictions_shape[0], predictions_shape[1], -1,
                          predictions_shape[-1]])
        labels_shape = common_layers.shape_list(labels)
        labels = tf.reshape(
            labels, [labels_shape[0], labels_shape[1], -1])
    return predictions, labels


def get_t2t_metric_op(metric_fn, predictions, features, labels,
                      weights_fn=common_layers.weights_nonzero):
    """Metric fn."""
    # Send along the entire features dict if the metric fn has the kwarg
    # "features".
    kwargs = {}
    args, _, keywords, _ = inspect.getargspec(metric_fn)
    if ("features" in args) or keywords:
        kwargs["features"] = features

    predictions, labels = reduce_dimensions(predictions, labels)

    scores, weights = metric_fn(predictions, labels,
                                weights_fn=weights_fn, **kwargs)
    return tf.metrics.mean(scores, weights)
