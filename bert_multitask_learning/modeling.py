from __future__ import absolute_import, division, print_function

import collections
import re

import six
import tensorflow as tf
import transformers

from bert_multitask_learning.params import BaseParams

from .utils import (get_embedding_table_from_model,
                    load_transformer_model)

LOGGER = tf.get_logger()


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1 - (1.0 - dropout_prob))
    return output


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.keras.layers.LayerNormalization(name=name)(input_tensor)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False,
                     embedding_table=None):
    """Looks up words embeddings for id tensor.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.nn.embedding_lookup()`. One hot is better
        for TPUs.

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    if embedding_table is None:
        embedding_table = tf.compat.v1.get_variable(
            name=word_embedding_name,
            shape=[vocab_size, embedding_size],
            initializer=create_initializer(initializer_range))

    if use_one_hot_embeddings:
        flat_input_ids = tf.reshape(input_ids, [-1])
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.nn.embedding_lookup(params=embedding_table, ids=input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(input=tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.compat.v1.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


class MultiModalBertModel(tf.keras.Model):
    def __init__(self, params: BaseParams, use_one_hot_embeddings=False):
        super(MultiModalBertModel, self).__init__()
        self.params = params
        if self.params.init_weight_from_huggingface:
            self.bert_model = load_transformer_model(
                self.params.transformer_model_name)
        else:
            self.bert_model = load_transformer_model(self.params.bert_config)
            self.bert_model(tf.convert_to_tensor(
                transformers.file_utils.DUMMY_INPUTS))
        self.use_one_hot_embeddings = use_one_hot_embeddings

        # multimodal input dense
        self.modal_name_list = ['image', 'others']
        self.multimodal_dense = {modal_name: tf.keras.layers.Dense(
            self.bert_model.config.hidden_size) for modal_name in self.modal_name_list}

        # multimodal modal type embedding
        # this might raise no gradients warning if it's unimodal
        # variable: [3, 768]
        if self.params.enable_modal_type:
            self.modal_type_embedding = tf.keras.layers.Embedding(input_dim=len(
                self.modal_name_list)+1, output_dim=self.bert_model.config.hidden_size)

        self.enable_modal_type = self.params.enable_modal_type

    def call(self, inputs, training):
        features_dict = inputs
        input_ids = features_dict['input_ids']
        input_mask = features_dict['input_mask']
        token_type_ids = features_dict['segment_ids']
        input_shape = get_shape_list(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(
                shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(
                shape=[batch_size, seq_length], dtype=tf.int32)

        config = self.bert_model.config

        self.embedding_output = tf.gather(
            get_embedding_table_from_model(self.bert_model), input_ids)

        # we need to add [SEP] embeddings around modal input
        # Since the last input_ids is always [SEP], we can use it directly
        sep_embedding = tf.expand_dims(
            self.embedding_output[:, -1, :], axis=1)

        if self.enable_modal_type:
            # for multimodal
            modal_type_ids = tf.zeros_like(input_ids)
        else:
            modal_type_ids = None

        for modal_name in self.modal_name_list:
            input_name = '{}_input'.format(modal_name)
            segment_id_name = '{}_segment_ids'.format(modal_name)
            mask_name = '{}_mask'.format(modal_name)
            if input_name not in features_dict:
                continue

            if not self.enable_modal_type:
                LOGGER.warning('Seems there\'s a multimodal inputs but params.enable_modal_type is '
                               'not set to be True.')

            # convert other modal embeddings to hidden_size
            # [batch_size, seq_length, modal_dim] -> [batch_size, seq_length, hidden_size]
            modal_input = self.multimodal_dense[modal_name](
                features_dict[input_name])

            # add sep embedding
            modal_input = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                [sep_embedding, modal_input, sep_embedding], axis=1)
            # add same type id to left and right
            modal_segment_ids = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                [tf.expand_dims(features_dict[segment_id_name][:, 0], axis=1),
                    features_dict[segment_id_name],
                    tf.expand_dims(features_dict[segment_id_name][:, 0], axis=1)], axis=1)
            # add mask
            modal_mask = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                [tf.expand_dims(features_dict[mask_name][:, 0], axis=1),
                    features_dict[mask_name],
                    tf.expand_dims(features_dict[mask_name][:, 0], axis=1)], axis=1)
            # add modal type
            if self.enable_modal_type:
                this_modal_type_ids = tf.ones_like(
                    modal_segment_ids) * self.params.modal_type_id[modal_name]

            # concat to text correspondingly
            self.embedding_output = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                [self.embedding_output, modal_input], axis=1)
            token_type_ids = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                [token_type_ids, modal_segment_ids], axis=1)
            input_mask = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                [input_mask, modal_mask], axis=1)
            if self.enable_modal_type:
                modal_type_ids = tf.concat(
                    [modal_type_ids, this_modal_type_ids], axis=1)

        self.model_input_mask = input_mask
        self.model_token_type_ids = token_type_ids
        if self.enable_modal_type:
            self.model_modal_type_ids = modal_type_ids

        word_embedding = self.embedding_output
        if self.enable_modal_type:
            word_embedding = word_embedding + \
                self.modal_type_embedding(modal_type_ids)

        outputs = self.bert_model(
            {'input_ids': None,
             'inputs_embeds': word_embedding,
             'attention_mask': input_mask,
             'token_type_ids': token_type_ids,
             'position_ids': input_mask},
            training=training,
            output_hidden_states=True,
            return_dict=True
        )
        self.sequence_output = outputs.last_hidden_state
        self.pooled_output = outputs.pooler_output
        self.all_encoder_layers = tf.stack(outputs.hidden_states, axis=1)
        return outputs

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the output of the embedding layer, after summing the word
          embeddings with the positional embeddings and the token type embeddings,
          then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return get_embedding_table_from_model(self.bert_model)

    def get_input_mask(self):
        return self.model_input_mask

    def get_token_type_ids(self):
        return self.model_token_type_ids


def get_assignment_map_from_keras_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)
    # convert init vars name

    def ckpt_name_to_train_name(var_name):
        # bert/encoder/layer/9/intermediate/dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
        tmp_name = var_name.replace(
            '/.ATTRIBUTES/VARIABLE_VALUE', '').replace('self_attention', 'self').replace('dense_output', 'output').replace('bert_output', 'output')
        return tmp_name.replace('layer/', 'layer_._')
    init_var_dict = {v: ckpt_name_to_train_name(v) for v, _ in init_vars}

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        graph_var_name = None
        ckpt_var_name = None
        for train_name in name_to_variable.keys():
            if init_var_dict[name] in train_name:
                graph_var_name = train_name
                ckpt_var_name = name
                break
        if graph_var_name and ckpt_var_name:
            assignment_map[ckpt_var_name] = name_to_variable[graph_var_name]
            initialized_variable_names[graph_var_name] = 1
            initialized_variable_names[graph_var_name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


class MultiModalPretrainInputs(tf.keras.Model):
    def __init__(self, params: BaseParams):
        super(MultiModalPretrainInputs, self).__init__()

    def call(self, input, training):
        pass
