import tensorflow as tf

from bert_multitask_learning.predefined_problems import *

from bert_multitask_learning import DynamicBatchSizeParams
from bert_multitask_learning import train_bert_multitask, train_eval_input_fn, BertMultiTask, EVAL
import os
from bert_multitask_learning import predict_input_fn
from bert_multitask_learning import eval_bert_multitask, predict_bert_multitask, trim_checkpoint_for_prediction

import shutil
import tempfile

tmptmpdir = tempfile.mkdtemp()
tmpckptdir = tempfile.mkdtemp()
tmptrimdir = tempfile.mkdtemp()

logger = tf.get_logger()
logger.setLevel('DEBUG')
logger.propagate = False

params = DynamicBatchSizeParams()
params.dynamic_padding = False
params.shuffle_buffer = 1000

# problem = 'city_cws|weibo_ner&weibo_cws'
problem = 'weibo_ner&weibo_fake_cls|weibo_fake_multi_cls|weibo_masklm'
# problem = 'weibo_ner|weibo_fake_cls'
num_gpus = 1
num_epochs = 1
model_dir = None
processing_fn_dict = None
problem_type_dict = None
params.log_every_n_steps = 1
params.bert_num_hidden_layer = 1
params.freeze_step = 0
params.mean_gradients = False
params.max_seq_len = 10
# params.transformer_config_loading = 'BertConfig'
params.transformer_tokenizer_loading = 'BertTokenizer'
params.transformer_model_loading = 'AlbertForMaskedLM'
params.transformer_config_loading = 'AlbertConfig'
params.transformer_model_name = 'voidful/albert_chinese_tiny'
params.transformer_config_name = 'voidful/albert_chinese_tiny'
params.transformer_tokenizer_name = 'voidful/albert_chinese_tiny'
params.tmp_file_dir = tmptmpdir
params.ckpt_dir = tmpckptdir

problem_type_dict = {
    'weibo_ner': 'seq_tag',
    'weibo_cws': 'seq_tag',
    'weibo_fake_multi_cls': 'multi_cls',
    'weibo_fake_cls': 'cls',
    'weibo_masklm': 'masklm'
}

processing_fn_dict = {
    'weibo_ner': get_weibo_ner_fn(file_path='/data/bert-multitask-learning/data/ner/weiboNER*'),
    'weibo_cws': get_weibo_cws_fn(file_path='/data/bert-multitask-learning/data/ner/weiboNER*'),
    'weibo_fake_cls': get_weibo_fake_cls_fn(file_path='/data/bert-multitask-learning/data/ner/weiboNER*'),
    'weibo_fake_multi_cls': get_weibo_fake_multi_cls_fn(file_path='/data/bert-multitask-learning/data/ner/weiboNER*'),
    'weibo_masklm': get_weibo_masklm(file_path='/data/bert-multitask-learning/data/ner/weiboNER*')
}

model = train_bert_multitask(
    problem=problem,
    num_epochs=1,
    params=params,
    problem_type_dict=problem_type_dict,
    processing_fn_dict=processing_fn_dict,
    steps_per_epoch=10,
    continue_training=True
)

model = train_bert_multitask(
    problem=problem,
    num_epochs=1,
    params=params,
    problem_type_dict=problem_type_dict,
    processing_fn_dict=processing_fn_dict,
    continue_training=True
)
trim_checkpoint_for_prediction(
    problem=problem, input_dir=model.params.ckpt_dir,
    output_dir=model.params.ckpt_dir+'_pred',
    problem_type_dict=problem_type_dict, overwrite=True)

# model_dir = params.ckpt_dir

model_dir = model.params.ckpt_dir

test_predict = ['这是一个测试样例']*10
params = DynamicBatchSizeParams()
params.transformer_tokenizer_loading = 'BertTokenizer'
params.transformer_model_loading = 'AlbertForMaskedLM'
params.transformer_config_loading = 'AlbertConfig'
params.transformer_model_name = 'voidful/albert_chinese_tiny'
params.transformer_config_name = 'voidful/albert_chinese_tiny'
params.transformer_tokenizer_name = 'voidful/albert_chinese_tiny'
pred, model = predict_bert_multitask(
    problem='weibo_ner',
    inputs=test_predict, model_dir=model_dir,
    problem_type_dict=problem_type_dict,
    processing_fn_dict=processing_fn_dict, return_model=True,
    params=params)
print(pred)
for p in pred:
    print(p)
# def pred_input_fn(): return predict_input_fn(
#     input_file_or_list=test_predict, config=params)
# pred = estimator.pred

eval_bert_multitask(problem=problem, params=params,
                    problem_type_dict=problem_type_dict, processing_fn_dict=processing_fn_dict,
                    model_dir=model_dir)
shutil.rmtree(tmptmpdir)
shutil.rmtree(tmpckptdir)
shutil.rmtree(tmptrimdir)
