# AUTOGENERATED! DO NOT EDIT! File to edit: source_nbs/00_1_params.ipynb (unless otherwise specified).

__all__ = ['Params']

# Cell

from .base_params import BaseParams
from .embedding_layer.base import (DefaultMultimodalEmbedding,
                                       DuplicateAugMultimodalEmbedding)
from .loss_strategy.base import SumLossCombination
from .mtl_model.mmoe import MMoE
from .problem_types import cls as problem_type_cls
from .problem_types import (contrastive_learning, masklm, multi_cls,
                                premask_mlm, pretrain, regression, seq_tag,
                                vector_fit)


class Params(BaseParams):
    def __init__(self):
        super().__init__()
        # register pre-defined problem types
        self.register_problem_type(problem_type='cls',
                                   top_layer=problem_type_cls.Classification,
                                   label_handling_fn=problem_type_cls.cls_label_handling_fn,
                                   get_or_make_label_encoder_fn=problem_type_cls.cls_get_or_make_label_encoder_fn,
                                   description='Classification')
        self.register_problem_type(problem_type='multi_cls',
                                   top_layer=multi_cls.MultiLabelClassification,
                                   label_handling_fn=multi_cls.multi_cls_label_handling_fn,
                                   get_or_make_label_encoder_fn=multi_cls.multi_cls_get_or_make_label_encoder_fn,
                                   description='Multi-Label Classification')
        self.register_problem_type(problem_type='seq_tag',
                                   top_layer=seq_tag.SequenceLabel,
                                   label_handling_fn=seq_tag.seq_tag_label_handling_fn,
                                   get_or_make_label_encoder_fn=seq_tag.seq_tag_get_or_make_label_encoder_fn,
                                   description='Sequence Labeling')
        self.register_problem_type(problem_type='masklm',
                                   top_layer=masklm.MaskLM,
                                   label_handling_fn=masklm.masklm_label_handling_fn,
                                   get_or_make_label_encoder_fn=masklm.masklm_get_or_make_label_encoder_fn,
                                   description='Masked Language Model')
        self.register_problem_type(problem_type='pretrain',
                                   top_layer=pretrain.PreTrain,
                                   label_handling_fn=pretrain.pretrain_label_handling_fn,
                                   get_or_make_label_encoder_fn=pretrain.pretrain_get_or_make_label_encoder_fn,
                                   description='NSP+MLM(Deprecated)')
        self.register_problem_type(problem_type='regression',
                                   top_layer=regression.Regression,
                                   label_handling_fn=regression.regression_label_handling_fn,
                                   get_or_make_label_encoder_fn=regression.regression_get_or_make_label_encoder_fn,
                                   description='Regression')
        self.register_problem_type(
            problem_type='vector_fit',
            top_layer=vector_fit.VectorFit,
            label_handling_fn=vector_fit.vector_fit_label_handling_fn,
            get_or_make_label_encoder_fn=vector_fit.vector_fit_get_or_make_label_encoder_fn,
            description='Vector Fitting')
        self.register_problem_type(
            problem_type='premask_mlm',
            top_layer=premask_mlm.PreMaskMLM,
            label_handling_fn=premask_mlm.premask_mlm_label_handling_fn,
            get_or_make_label_encoder_fn=premask_mlm.premask_mlm_get_or_make_label_encoder_fn,
            description='Pre-masked Masked Language Model'
        )
        self.register_problem_type(
            problem_type='contrastive_learning',
            top_layer=contrastive_learning.ContrastiveLearning,
            label_handling_fn=contrastive_learning.contrastive_learning_label_handling_fn,
            get_or_make_label_encoder_fn=contrastive_learning.contrastive_learning_get_or_make_label_encoder_fn,
            description='Contrastive Learning'
        )

        self.register_mtl_model(
            'mmoe', MMoE, include_top=False, extra_info='MMoE')
        self.register_loss_combination_strategy('sum', SumLossCombination)
        self.register_embedding_layer(
            'duplicate_data_augmentation_embedding', DuplicateAugMultimodalEmbedding)
        self.register_embedding_layer(
            'default_embedding', DefaultMultimodalEmbedding)

        self.assign_loss_combination_strategy('sum')
        self.assign_data_sampling_strategy()
        self.assign_embedding_layer('default_embedding')
