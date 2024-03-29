# AUTOGENERATED! DO NOT EDIT! File to edit: source_nbs/12_0_problem_type_utils.ipynb (unless otherwise specified).

__all__ = ['empty_tensor_handling_loss', 'nan_loss_handling', 'create_dummy_if_empty', 'BaseTop', 'pad_to_shape']

# Cell
from typing import Dict, Tuple

import tensorflow as tf
from ..base_params import BaseParams


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


@tf.function
def nan_loss_handling(loss):
    if tf.math.is_nan(loss):
        return 0.0
    else:
        return loss


@tf.function
def create_dummy_if_empty(inp_tensor: tf.Tensor) -> tf.Tensor:
    shape_tensor = tf.shape(inp_tensor)
    if tf.equal(shape_tensor[0], 0):
        data_type = inp_tensor.dtype
        dummy_shape_first_dim = tf.convert_to_tensor([1], dtype=tf.int32)
        dummy_shape = tf.concat(
            [dummy_shape_first_dim, shape_tensor[1:]], axis=0)
        dummy_tensor = tf.zeros(dummy_shape, dtype=data_type)
        return dummy_tensor
    else:
        return inp_tensor


class BaseTop(tf.keras.Model):
    def __init__(self, params: BaseParams, problem_name: str) -> None:
        super(BaseTop, self).__init__(name=problem_name)
        self.params = params
        self.problem_name = problem_name

    def call(self, inputs: Tuple[Dict], mode: str):
        raise NotImplementedError

def pad_to_shape(from_tensor: tf.Tensor, to_tensor: tf.Tensor, axis=1) -> tf.Tensor:
    # sometimes the length of labels dose not equal to length of inputs
    # that's caused by tf.data.experimental.bucket_by_sequence_length in multi problem scenario
    pad_len = tf.shape(input=to_tensor)[
        axis] - tf.shape(input=from_tensor)[axis]

    # top, bottom, left, right
    pad_tensor = [[0, 0] for _ in range(len(from_tensor.shape))]
    pad_tensor[axis] = [0, pad_len]
    from_tensor = tf.pad(tensor=from_tensor, paddings=pad_tensor)
    return from_tensor
