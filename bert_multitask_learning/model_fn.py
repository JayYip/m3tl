
import tensorflow as tf

from . import modeling
from .modeling import MultiModalBertModel
from .optimizer import AdamWeightDecayOptimizer
from .top import (Classification, MaskLM, MultiLabelClassification, PreTrain,
                  Seq2Seq, SequenceLabel)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.compat.v1.name_scope(name):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(
                input_tensor=tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
        tf.compat.v1.summary.histogram('histogram', var)


@tf.function
def filter_loss(loss, features, problem):

    if tf.reduce_mean(input_tensor=features['%s_loss_multiplier' % problem]) == 0:
        return_loss = 0.0
    else:
        return_loss = loss

    return return_loss


class BertMultiTask():
    """Main model class for creating Bert multi-task model

    Method:
        body: takes input_ids, segment_ids, input_mask and pass through bert model
        top: takes the logit output from body and apply classification top layer
        create_spec: create train, eval and predict EstimatorSpec
        get_model_fn: get bert multi-task model function
    """

    def __init__(self, params):
        self.params = params

    def body(self, features, mode):
        """Body of the model, aka Bert

        Arguments:
            features {dict} -- feature dict,
                keys: input_ids, input_mask, segment_ids
            mode {mode} -- mode

        Returns:
            dict -- features extracted from bert.
                keys: 'seq', 'pooled', 'all', 'embed'

        seq:
            tensor, [batch_size, seq_length, hidden_size]
        pooled:
            tensor, [batch_size, hidden_size]
        all:
            list of tensor, num_hidden_layers * [batch_size, seq_length, hidden_size]
        embed:
            tensor, [batch_size, seq_length, hidden_size]
        """

        config = self.params
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = MultiModalBertModel(
            params=self.params,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=config.use_one_hot_embeddings,
            features_dict=features)
        self.model = model
        features['model_input_mask'] = self.model.get_input_mask()
        features['model_token_type_ids'] = self.model.get_token_type_ids()

        feature_dict = {}
        for logit_type in ['seq', 'pooled', 'all', 'embed', 'embed_table']:
            if logit_type == 'seq':
                # tensor, [batch_size, seq_length, hidden_size]
                feature_dict[logit_type] = model.get_sequence_output()
            elif logit_type == 'pooled':
                # tensor, [batch_size, hidden_size]
                feature_dict[logit_type] = model.get_pooled_output()
            elif logit_type == 'all':
                # list, num_hidden_layers * [batch_size, seq_length, hidden_size]
                feature_dict[logit_type] = model.get_all_encoder_layers()
            elif logit_type == 'embed':
                # for res connection
                feature_dict[logit_type] = model.get_embedding_output()
            elif logit_type == 'embed_table':
                feature_dict[logit_type] = model.get_embedding_table()

        # add summary
        if self.params.detail_log:
            with tf.compat.v1.name_scope('bert_feature_summary'):
                for _, layer_output in enumerate(feature_dict['all']):
                    variable_summaries(
                        layer_output, layer_output.name.replace(':0', ''))

        feature_dict['all'] = tf.concat(feature_dict['all'], axis=1)

        return feature_dict

    def get_features_for_problem(self, features, hidden_feature, problem, mode):
        # get features with ind == 1
        if mode == tf.estimator.ModeKeys.PREDICT:
            feature_this_round = features
            hidden_feature_this_round = hidden_feature
        else:
            multiplier_name = '%s_loss_multiplier' % problem

            record_ind = tf.compat.v1.where(tf.cast(
                features[multiplier_name], tf.bool))

            hidden_feature_this_round = {}
            for hidden_feature_name in hidden_feature:
                if hidden_feature_name != 'embed_table':
                    hidden_feature_this_round[hidden_feature_name] = tf.gather_nd(
                        hidden_feature[hidden_feature_name], record_ind
                    )
                else:
                    hidden_feature_this_round[hidden_feature_name] = hidden_feature[hidden_feature_name]

            feature_this_round = {}
            for features_name in features:
                feature_this_round[features_name] = tf.gather_nd(
                    features[features_name],
                    record_ind)

        return feature_this_round, hidden_feature_this_round

    def hidden(self, features, hidden_feature, mode):
        """Hidden of model, will be called between body and top

        This is majorly for all the crazy stuff.

        Arguments:
            features {dict of tensor} -- feature dict
            hidden_feature {dict of tensor} -- hidden feature dict output by body
            mode {mode} -- ModeKey

        Raises:
            ValueError: Incompatible submodels

        Returns:
            return_feature
            return_hidden_feature
        """

        if self.params.label_transfer:
            raise NotImplementedError
            # ori_hidden_feature = {
            #     'ori_'+k: v for k,
            #     v in hidden_feature.items()}
            # label_transfer_layer = LabelTransferHidden(self.params)
            # hidden_feature = label_transfer_layer(
            #     features, hidden_feature, mode)
            # hidden_feature.update(ori_hidden_feature)

        if self.params.task_transformer:
            raise NotImplementedError
            # task_tranformer_layer = TaskTransformer(self.params)
            # task_tranformer_hidden_feature = task_tranformer_layer(
            #     features, hidden_feature, mode)
            # self.params.hidden_dense = False

        return_feature = {}
        return_hidden_feature = {}

        for problem_dict in self.params.run_problem_list:
            for problem in problem_dict:
                if self.params.task_transformer:
                    hidden_feature = task_tranformer_hidden_feature[problem]
                problem_type = self.params.problem_type[problem]

                top_scope_name = self.get_scope_name(problem)

                if len(self.params.run_problem_list) > 1:
                    feature_this_round, hidden_feature_this_round = self.get_features_for_problem(
                        features, hidden_feature, problem, mode)
                else:
                    feature_this_round, hidden_feature_this_round = features, hidden_feature

                if self.params.label_transfer and self.params.grid_transformer:
                    raise ValueError(
                        'Label Transfer and grid transformer cannot be enabled in the same time.'
                    )

                if self.params.grid_transformer:
                    raise NotImplementedError
                    # with tf.compat.v1.variable_scope(top_scope_name):
                    #     grid_layer = GridTransformer(self.params)

                    #     hidden_feature_key = 'pooled' if problem_type == 'cls' else 'seq'

                    #     hidden_feature_this_round[hidden_feature_key] = grid_layer(
                    #         feature_this_round, hidden_feature_this_round, mode, problem)
                    # self.params.hidden_dense = False
                return_hidden_feature[problem] = hidden_feature_this_round
                return_feature[problem] = feature_this_round
        return return_feature, return_hidden_feature

    def top(self, features, hidden_feature, mode):
        """Top model. This fn will return:
        1. loss, if mode is train
        2, eval_metric, if mode is eval
        3, prob, if mode is pred

        Arguments:
            features {dict} -- feature dict
            hidden_feature {dict} -- hidden feature dict extracted by bert
            mode {mode key} -- mode

        """
        problem_type_layer = {
            'seq_tag': SequenceLabel,
            'cls': Classification,
            'seq2seq_tag': Seq2Seq,
            'seq2seq_text': Seq2Seq,
            'multi_cls': MultiLabelClassification
        }

        return_dict = {}

        for problem_dict in self.params.run_problem_list:
            for problem in problem_dict:
                feature_this_round = features[problem]
                hidden_feature_this_round = hidden_feature[problem]
                problem_type = self.params.problem_type[problem]
                scope_name = self.params.share_top[problem]

                top_scope_name = self.get_scope_name(problem)

                # if pretrain, return pretrain logit
                if problem_type == 'pretrain':
                    pretrain = PreTrain(self.params)
                    return_dict[scope_name] = pretrain(
                        feature_this_round, hidden_feature_this_round, mode, problem)
                    return return_dict

                if self.params.label_transfer and self.params.grid_transformer:
                    raise ValueError(
                        'Label Transfer and grid transformer cannot be enabled in the same time.'
                    )

                with tf.compat.v1.variable_scope(top_scope_name, reuse=tf.compat.v1.AUTO_REUSE):
                    layer = problem_type_layer[
                        problem_type](self.params)
                    return_dict[problem] = layer(
                        feature_this_round,
                        hidden_feature_this_round, mode, problem)

                    if mode == tf.estimator.ModeKeys.TRAIN:
                        return_dict[problem] = filter_loss(
                            return_dict[problem], feature_this_round, problem)

        if self.params.augument_mask_lm and mode == tf.estimator.ModeKeys.TRAIN:
            try:
                mask_lm_top = MaskLM(self.params)
                return_dict['augument_mask_lm'] = \
                    mask_lm_top(features,
                                hidden_feature, mode, 'dummy')
            except ValueError:
                pass

        return return_dict

    def get_scope_name(self, problem):
        scope_name = self.params.share_top[problem]
        top_scope_name = '%s_top' % scope_name
        if self.params.task_transformer:
            top_scope_name = top_scope_name + '_after_tr'

        if self.params.label_transfer:
            top_scope_name = top_scope_name + '_lt'
            if self.params.hidden_gru:
                top_scope_name += '_gru'
        return top_scope_name

    def create_optimizer(self, init_lr, num_train_steps, num_warmup_steps):
        """Creates an optimizer training op."""
        global_step = tf.compat.v1.train.get_or_create_global_step()

        learning_rate = tf.constant(
            value=init_lr, shape=[], dtype=tf.float32)

        # Implements linear decay of the learning rate.
        learning_rate = tf.compat.v1.train.polynomial_decay(
            learning_rate,
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)

        # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
        # learning rate will be `global_step/num_warmup_steps * init_lr`.
        if num_warmup_steps:
            global_steps_int = tf.cast(global_step, tf.int32)
            warmup_steps_int = tf.constant(
                num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = init_lr * warmup_percent_done

            is_warmup = tf.cast(global_steps_int <
                                warmup_steps_int, tf.float32)
            learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

        tf.compat.v1.summary.scalar('lr', learning_rate)

        self.learning_rate = learning_rate

        # It is recommended that you use this optimizer for fine tuning, since this
        # is how the model was trained (note that the Adam m/v variables are NOT
        # loaded from init_checkpoint.)
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        return optimizer

    def create_train_spec(self, loss_eval_pred, mode, scaffold_fn):
        optimizer = self.create_optimizer(
            self.params.lr,
            self.params.train_steps,
            self.params.num_warmup_steps)

        global_step = tf.compat.v1.train.get_or_create_global_step()

        tvars = tf.compat.v1.trainable_variables()

        total_loss = 0
        hook_dict = {}
        for k, l in loss_eval_pred.items():
            hook_dict['%s_loss' % k] = l
            total_loss += l

        hook_dict['learning_rate'] = self.learning_rate
        hook_dict['total_training_steps'] = tf.constant(
            self.params.train_steps)

        logging_hook = tf.estimator.LoggingTensorHook(
            hook_dict, every_n_iter=self.params.log_every_n_steps)

        grads = tf.gradients(
            ys=total_loss, xs=tvars,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        if self.params.mean_gradients:
            for v_idx, v in enumerate(tvars):
                if v.name.startswith('bert/'):
                    grads[v_idx] = grads[v_idx] / \
                        len(self.params.run_problem_list)

        if self.params.freeze_step > 0:
            # if global_step > freeze_step, gradient_mask == 1, inverse_gradient_mask == 0
            # else: reverse
            gradient_mask = tf.cast(tf.greater(
                global_step, self.params.freeze_step), dtype=tf.float32)
            for v_idx, v in enumerate(tvars):
                if v.name.startswith('bert/'):
                    grads[v_idx] = grads[v_idx] * gradient_mask

        if self.params.detail_log:
            # add grad summary
            with tf.compat.v1.name_scope('var_and_grads'):
                for g, v in zip(grads, tvars):
                    if g is not None:
                        variable_summaries(g, v.name.replace(':0', '-grad'))
                        variable_summaries(v, v.name.replace(':0', ''))

        # This is how the model was pre-trained.
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

        train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=global_step)

        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            training_hooks=[logging_hook],
            scaffold=scaffold_fn)
        return output_spec

    def create_spec(self, features, loss_eval_pred, mode, warm_start):
        """Function to create spec for different mode

        Arguments:
            features {dict} -- feature dict
            hidden_features {dict} -- hidden feature dict extracted by bert
            loss_eval_pred {None} -- see self.top
            mode {mode} -- mode

        Returns:
            spec -- train\eval\predict spec
        """

        tvars = tf.compat.v1.trainable_variables()

        if mode == tf.estimator.ModeKeys.TRAIN:
            init_decoder = False
            if self.params.init_weight_from_huggingface:
                # ckpt_path = os.path.join(self.params.transformer_model_name,
                #                          'pretrained_model')
                ckpt_path = self.params.transformer_model_name
                (assignment_map, _
                 ) = modeling.get_assignment_map_from_keras_checkpoint(tvars, ckpt_path)

                # init decoder weight
                if self.params.transformer_decoder_model_name:
                    (decoder_assignment_map, _
                     ) = modeling.get_assignment_map_from_keras_checkpoint(tvars, self.params.transformer_decoder_model_name)
                    init_decoder = True

            else:
                ckpt_path = self.params.init_checkpoint

                (assignment_map, _
                 ) = modeling.get_assignment_map_from_checkpoint(tvars, ckpt_path)

            def scaffold():
                init_op = tf.compat.v1.train.init_from_checkpoint(
                    ckpt_path, assignment_map)
                if init_decoder:
                    decoder_init_op = tf.compat.v1.train.init_from_checkpoint(
                        self.params.transformer_decoder_model_name, decoder_assignment_map)
                    init_op = tf.group(init_op, decoder_init_op)
                return tf.compat.v1.train.Scaffold(init_op)

            if not warm_start:
                train_scaffold = None
            else:
                if not assignment_map:
                    raise ValueError(
                        'No variable initialized from pretrained checkpoint!')
                train_scaffold = scaffold()

            return self.create_train_spec(loss_eval_pred,
                                          mode,
                                          train_scaffold)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_loss = {k: v[1] for k, v in loss_eval_pred.items()}
            total_eval_loss = 0
            for _, var in eval_loss.items():
                total_eval_loss += var

            eval_metric = {k: v[0] for k, v in loss_eval_pred.items()}
            total_eval_metric = {}
            for problem, metric in eval_metric.items():
                for metric_name, metric_tuple in metric.items():
                    total_eval_metric['%s_%s' %
                                      (problem, metric_name)] = metric_tuple

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_eval_loss,
                eval_metric_ops=total_eval_metric)
            return output_spec
        else:
            # include input ids
            loss_eval_pred['input_ids'] = features['input_ids']
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=loss_eval_pred)
            return output_spec

    def get_model_fn(self, warm_start=False):
        def model_fn(features, labels, mode, params):

            hidden_feature = self.body(
                features, mode)

            problem_sep_features, hidden_feature = self.hidden(
                features, hidden_feature, mode)

            loss_eval_pred = self.top(
                problem_sep_features, hidden_feature, mode)

            spec = self.create_spec(
                features, loss_eval_pred, mode, warm_start)
            return spec

        return model_fn
