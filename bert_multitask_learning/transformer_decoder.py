import tensorflow as tf
import math

from . import modeling


class TransformerDecoder(object):
    def __init__(self, params):
        self.params = params

    def get_decoder_self_attention_mask(self, length):
        """Calculate bias for decoder that maintains model's autoregressive property.
        Creates a tensor that masks out locations that correspond to illegal
        connections, so prediction at position i cannot draw information from future
        positions.
        Args:
            length: int length of sequences in batch.
        Returns:
            float tensor of shape [1, 1, length, length]
        """
        with tf.name_scope("decoder_self_attention_mask"):
            valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
            valid_locs = tf.reshape(valid_locs, [1, length, length])
        return valid_locs

    def decode(
            self,
            decoder_inputs,
            encoder_output,
            input_mask,
            decoder_self_attention_mask,
            cache,
            num_classes,
            do_return_all_layers,
            enc_dec_attention_mask=None,
            add_self_attention=True,
            add_enc_dec_attention=True):
        input_tensor = decoder_inputs
        num_hidden_layers = self.params.decoder_num_hidden_layers
        hidden_size = self.params.bert_config.hidden_size
        num_attention_heads = self.params.bert_config.num_attention_heads
        initializer_range = self.params.bert_config.initializer_range
        attention_probs_dropout_prob = self.params.bert_config.attention_probs_dropout_prob

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        attention_head_size = int(hidden_size / num_attention_heads)
        encode_shape = modeling.get_shape_list(
            encoder_output, expected_rank=3)
        batch_size = encode_shape[0]
        encode_seq_length = encode_shape[1]
        input_width = encode_shape[2]

        input_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
        decode_seq_length = input_shape[1]

        # create encoder-decoder attention mask
        attention_mask_shape = modeling.get_shape_list(
            input_mask, expected_rank=2)[1]

        # batch_size*beam_size
        if enc_dec_attention_mask is None:
            input_batch_size = modeling.get_shape_list(
                decoder_inputs, expected_rank=3)[0]
            input_mask = tf.broadcast_to(
                input_mask, [input_batch_size, attention_mask_shape])
            attention_mask = modeling.create_attention_mask_from_input_mask(
                decoder_inputs, input_mask
            )
        else:
            attention_mask = enc_dec_attention_mask

        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != hidden_size:
            raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                             (input_width, hidden_size))

        prev_output = modeling.reshape_to_matrix(input_tensor)

        all_layer_outputs = []
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output

                if cache is not None:
                    layer_cache = cache[str(layer_idx)]
                    if layer_idx == 0:
                        layer_input = tf.expand_dims(
                            layer_input, axis=1)
                        # update batch_size to batch_size*beam_size
                        batch_size = modeling.get_shape_list(
                            layer_input, expected_rank=3)[0]
                else:
                    layer_cache = None

                with tf.variable_scope("attention"):
                    attention_heads = []
                    if add_self_attention:
                        with tf.variable_scope("self"):
                            attention_head = attention_layer_with_cache(
                                from_tensor=layer_input,
                                to_tensor=layer_input,
                                attention_mask=decoder_self_attention_mask,
                                num_attention_heads=num_attention_heads,
                                size_per_head=attention_head_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                initializer_range=initializer_range,
                                do_return_2d_tensor=False,
                                batch_size=batch_size,
                                from_seq_length=decode_seq_length,
                                to_seq_length=decode_seq_length,
                                cache=layer_cache)
                            attention_heads.append(attention_head)

                            self_attention_output = None
                            if len(attention_heads) == 1:
                                self_attention_output = attention_heads[0]
                            else:
                                # In the case where we have other sequences, we just concatenate
                                # them to the self-attention head before the projection.
                                self_attention_output = tf.concat(
                                    attention_heads, axis=-1)
                        if cache is not None:
                            self_attention_output = tf.reshape(
                                self_attention_output, [batch_size, -1, hidden_size])
                    else:
                        self_attention_output = tf.reshape(
                            layer_input, [batch_size, -1, hidden_size])

                    if add_enc_dec_attention:
                        with tf.variable_scope('enc_dec_attention'):
                            attention_heads = []
                            attention_head = attention_layer_with_cache(
                                from_tensor=self_attention_output,
                                to_tensor=encoder_output,
                                attention_mask=attention_mask,
                                num_attention_heads=num_attention_heads,
                                size_per_head=attention_head_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                initializer_range=initializer_range,
                                do_return_2d_tensor=True,
                                batch_size=batch_size,
                                from_seq_length=decode_seq_length,
                                to_seq_length=encode_seq_length,
                                cache=None)
                            attention_heads.append(attention_head)

                            attention_output = None
                            if len(attention_heads) == 1:
                                attention_output = attention_heads[0]
                            else:
                                # In the case where we have other sequences, we just concatenate
                                # them to the self-attention head before the projection.
                                attention_output = tf.concat(
                                    attention_heads, axis=-1)
                        if cache is not None:
                            attention_output = tf.reshape(
                                attention_output, [batch_size, -1, hidden_size])
                    else:
                        attention_output = tf.reshape(
                            self_attention_output, [-1, hidden_size])

                    # Run a linear projection of `hidden_size` then add a residual
                    # with `layer_input`.
                    with tf.variable_scope("output"):
                        attention_output = tf.layers.dense(
                            attention_output,
                            hidden_size,
                            kernel_initializer=modeling.create_initializer(
                                initializer_range))
                        attention_output = modeling.dropout(
                            attention_output,
                            self.params.bert_config.hidden_dropout_prob)
                        attention_output = modeling.layer_norm(
                            attention_output + layer_input)

                # The activation is only applied to the "intermediate" hidden layer.
                with tf.variable_scope("intermediate"):
                    intermediate_output = tf.layers.dense(
                        attention_output,
                        self.params.bert_config.intermediate_size,
                        activation=modeling.gelu,
                        kernel_initializer=modeling.create_initializer(
                            initializer_range))

                # Down-project back to `hidden_size` then add the residual.
                with tf.variable_scope("output"):
                    layer_output = tf.layers.dense(
                        intermediate_output,
                        hidden_size,
                        kernel_initializer=modeling.create_initializer(
                            initializer_range))
                    layer_output = modeling.dropout(
                        layer_output,
                        self.params.bert_config.hidden_dropout_prob)
                    layer_output = modeling.layer_norm(
                        layer_output + attention_output)
                    prev_output = layer_output
                    all_layer_outputs.append(layer_output)

        if do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = modeling.reshape_from_matrix(
                    layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            if cache is None:
                final_output = modeling.reshape_from_matrix(
                    prev_output, input_shape)
            else:
                final_output = prev_output

        if num_classes:
            dense_layer = tf.layers.Dense(
                num_classes,
                activation=None,
                kernel_initializer=tf.orthogonal_initializer()
            )
            logits = dense_layer(final_output)
        else:
            logits = final_output
        return logits

    def train_eval(self, features, hidden_feature, mode, problem_name):

        # prepare inputs to attention
        key = 'ori_seq' if self.params.label_transfer else 'seq'
        encoder_output = hidden_feature[key]

        label_ids = features['%s_label_ids' % problem_name]
        input_mask = features['input_mask']
        num_classes = self.params.num_classes[problem_name]

        if self.params.problem_type[problem_name] == 'seq2seq_text':
            embed_table = hidden_feature['embed_table']
        else:
            embed_table = tf.get_variable(
                'tag_embed_table', shape=[
                    num_classes, self.params.mask_lm_hidden_size],
                initializer=tf.orthogonal_initializer())
        decoder_inputs = tf.nn.embedding_lookup(
            embed_table, label_ids)

        # with tf.name_scope("shift_targets"):
        #     # Shift targets to the right, and remove the last element
        #     decoder_inputs = tf.pad(
        #         decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

        decoder_inputs = modeling.embedding_postprocessor(
            input_tensor=decoder_inputs,
            use_token_type=False,
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=self.params.bert_config.initializer_range,
            max_position_embeddings=self.params.bert_config.max_position_embeddings,
            dropout_prob=self.params.bert_config.hidden_dropout_prob)

        # attention_mask = modeling.create_attention_mask_from_input_mask(
        #     label_ids, input_mask)
        label_mask = tf.expand_dims(
            tf.cast(features['%s_mask' % problem_name], tf.float32), axis=1)
        decoder_self_attention_mask = label_mask * self.get_decoder_self_attention_mask(
            self.params.decode_max_seq_len)

        decode_output = self.decode(
            decoder_inputs=decoder_inputs,
            encoder_output=encoder_output,
            input_mask=input_mask,
            decoder_self_attention_mask=decoder_self_attention_mask,
            cache=None,
            num_classes=num_classes,
            do_return_all_layers=False
        )
        return decode_output


def attention_layer_with_cache(from_tensor,
                               to_tensor,
                               attention_mask=None,
                               num_attention_heads=1,
                               size_per_head=512,
                               query_act=None,
                               key_act=None,
                               value_act=None,
                               attention_probs_dropout_prob=0.0,
                               initializer_range=0.02,
                               do_return_2d_tensor=False,
                               batch_size=None,
                               from_seq_length=None,
                               to_seq_length=None,
                               decoder_self_attention_mask=None,
                               cache=None):
    """
    This is a modification of attention layer from bert to support
    fast decode.

    Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      initializer_range: float. Range of the weight initializer.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.

    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = modeling.get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = modeling.get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = modeling.reshape_to_matrix(from_tensor)
    to_tensor_2d = modeling.reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=modeling.create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=modeling.create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=modeling.create_initializer(initializer_range))

    if cache is not None:
        n_time_h = key_layer.get_shape()[1]

        key_layer_to_cache = tf.reshape(
            key_layer, [batch_size, -1, n_time_h])
        value_layer_to_cache = tf.reshape(
            value_layer, [batch_size, -1, n_time_h])
        # Combine cached keys and values with new keys and values.
        key_layer_from_cache = tf.concat(
            [cache["key_layer"], key_layer_to_cache], axis=1)
        value_layer_from_cache = tf.concat(
            [cache["value_layer"], value_layer_to_cache], axis=1)

        # update seq length
        # from_seq_length = key_layer_from_cache.get_shape()[1]
        from_seq_length = modeling.get_shape_list(
            key_layer_from_cache, expected_rank=[3])[1]
        to_seq_length = modeling.get_shape_list(
            value_layer_from_cache, expected_rank=[3])[1]

        # Update cache
        cache["key_layer"] = key_layer_from_cache
        cache["value_layer"] = value_layer_from_cache

        key_layer = tf.reshape(key_layer_from_cache, [-1, n_time_h])
        value_layer = tf.reshape(value_layer_from_cache, [-1, n_time_h])

    # `query_layer` = [B, N, F, H]
    # In self attention of decoder, the seq_length of q always be 1
    if cache is not None:
        query_layer = transpose_for_scores(
            query_layer, batch_size,
            num_attention_heads, 1,
            size_per_head)
    else:
        query_layer = transpose_for_scores(
            query_layer, batch_size,
            num_attention_heads, from_seq_length,
            size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(
        key_layer, batch_size, num_attention_heads,
        to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = modeling.dropout(
        attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer
