# AUTOGENERATED! DO NOT EDIT! File to edit: source_nbs/12_10_problem_type_contrast_learning.ipynb (unless otherwise specified).

__all__ = ['SimCSE', 'get_contrastive_learning_model', 'ContrastiveLearning',
           'contrastive_learning_get_or_make_label_encoder_fn', 'contrastive_learning_label_handling_fn']

# Cell
from typing import List

import numpy as np
import tensorflow as tf
from loguru import logger
from ..base_params import BaseParams
from .utils import (empty_tensor_handling_loss,
                                      nan_loss_handling, pad_to_shape)
from ..special_tokens import PREDICT
from ..utils import (LabelEncoder, get_label_encoder_save_path, get_phase,
                        need_make_label_encoder)


# Cell
# export
class SimCSE(tf.keras.Model):
    def __init__(self, params: BaseParams, problem_name: str) -> None:
        super(SimCSE, self).__init__(name='simcse')
        self.params = params
        self.problem_name = problem_name
        self.dropout = tf.keras.layers.Dropout(self.params.dropout)
        self.pooler = self.params.get('simcse_pooler', 'pooled')
        self.metric_fn = tf.keras.metrics.CategoricalAccuracy(name='{}_acc'.format(problem_name))
        availabel_pooler = ['pooled', 'mean_pool']
        assert self.pooler in availabel_pooler, \
            'available params.simcse_pooler: {}, got: {}'.format(
                availabel_pooler, self.pooler)
        if self.params.embedding_layer['name'] != 'duplicate_data_augmentation_embedding':
            raise ValueError(
                'SimCSE requires duplicate_data_augmentation_embedding. Fix it with `params.assign_embedding_layer(\'duplicate_data_augmentation_embedding\')`')

    def call(self, inputs):

        features, hidden_features = inputs
        phase = get_phase()

        if phase != PREDICT:
            # created pool embedding
            if self.pooler == 'pooled':
                all_pooled_embedding = hidden_features['pooled']
            else:
                all_pooled_embedding = tf.reduce_mean(
                    hidden_features['seq'], axis=1)

            # shape (batch_size, hidden_dim)
            pooled_rep1_embedding, pooled_rep2_embedding = tf.split(
                all_pooled_embedding, 2)

            # calculate similarity
            pooled_rep1_embedding = tf.math.l2_normalize(
                pooled_rep1_embedding, axis=1)
            pooled_rep2_embedding = tf.math.l2_normalize(
                pooled_rep2_embedding, axis=1)
            # shape (batch_size, batch_size)
            similarity = tf.matmul(pooled_rep1_embedding,
                                   pooled_rep2_embedding, transpose_b=True)
            labels = tf.eye(tf.shape(similarity)[0])

            # shape (batch_size*batch_size)
            similarity = tf.reshape(similarity, shape=(-1, 1))
            labels = tf.reshape(labels, shape=(-1, 1))

            # make compatible with binary crossentropy
            similarity = tf.concat([1-similarity, similarity], axis=1)
            labels = tf.concat([1-labels, labels], axis=1)
            loss = tf.keras.losses.binary_crossentropy(labels, similarity)
            loss = tf.reduce_mean(loss)
            self.add_loss(loss)
            acc = self.metric_fn(labels, similarity)
            self.add_metric(acc)
        return inputs[1]['pooled']


# Cell
def get_contrastive_learning_model(params: BaseParams, problem_name: str, model_name: str) -> tf.keras.Model:
    if model_name == 'simcse':
        return SimCSE(params=params, problem_name=problem_name)

    logger.warning(
        '{} not match any contrastive learning model, using SimCSE'.format(model_name))
    return SimCSE(params=params, problem_name=problem_name)


# Cell

class ContrastiveLearning(tf.keras.Model):
    def __init__(self, params: BaseParams, problem_name: str) -> None:
        super(ContrastiveLearning, self).__init__(name=problem_name)
        self.params = params
        self.problem_name = problem_name
        self.contrastive_learning_model_name = self.params.contrastive_learning_model_name
        self.contrastive_learning_model = get_contrastive_learning_model(
            params=self.params, problem_name=problem_name, model_name=self.contrastive_learning_model_name)

    def call(self, inputs):
        return self.contrastive_learning_model(inputs)


# Cell
def contrastive_learning_get_or_make_label_encoder_fn(params: BaseParams, problem: str, mode: str, label_list: List[str], *args, **kwargs) -> LabelEncoder:

    le_path = get_label_encoder_save_path(params=params, problem=problem)
    label_encoder = LabelEncoder()
    if need_make_label_encoder(mode=mode, le_path=le_path, overwrite=kwargs['overwrite']):
        # fit and save label encoder
        label_encoder.fit(label_list)
        label_encoder.dump(le_path)
        params.set_problem_info(
            problem=problem, info_name='num_classes', info=len(label_encoder.encode_dict))
    else:
        label_encoder.load(le_path)

    return label_encoder


# Cell
def contrastive_learning_label_handling_fn(target: str, label_encoder=None, tokenizer=None, decoding_length=None, *args, **kwargs) -> dict:

    label_id = label_encoder.transform([target]).tolist()[0]
    label_id = np.int32(label_id)
    return label_id, None
