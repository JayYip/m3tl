import time
from collections import defaultdict
import os

import tensorflow as tf

from src.input_fn import train_eval_input_fn, predict_input_fn
from src.metrics import ner_evaluate
from src.model_fn import BertMultiTask
from src.params import Params
from src.utils import create_path
from src.estimator import Estimator
from src.ckpt_restore_hook import RestoreCheckpointHook


EXPERIMENTS_LIST = [
    {'baseline': {'problems': ['ascws', 'msrcws', 'pkucws',
                  'cityucws', 'WeiboNER', 'bosonner', 'msraner',
                  'CTBCWS', 'CTBPOS'],
                  'additional_params': {}},
     'mix_baseline': {
         'problems': ['CWS', 'NER', 'POS'],
         'additional_params': {}
     },
     'multitask_baseline': {
         'problems': ['CWS|NER|POS'],
         'additional_params': {}
     }
]

def train_problem(problem, gpu=4):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    params = Params()
    params.assign_problem(problem, gpu=int(gpu))

    create_path(params.ckpt_dir)

    tf.logging.info('Checkpoint dir: %s' % params.ckpt_dir)
    time.sleep(3)

    model = BertMultiTask(params=params)
    model_fn = model.get_model_fn(warm_start=False)

    dist_trategy = tf.contrib.distribute.MirroredStrategy(
        num_gpus=int(gpu),
        cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
            'nccl', num_packs=int(gpu)))

    run_config = tf.estimator.RunConfig(
        train_distribute=dist_trategy,
        eval_distribute=dist_trategy,
        log_step_count_steps=params.log_every_n_steps)

    # ws = make_warm_start_setting(params)

    estimator = Estimator(
        model_fn,
        model_dir=params.ckpt_dir,
        params=params,
        config=run_config)
    train_hook = RestoreCheckpointHook(params)

    def train_input_fn(): return train_eval_input_fn(params)
    estimator.train(
        train_input_fn, max_steps=params.train_steps, hooks=[train_hook])

    return estimator

def eval_problem(problem, estimator, gpu=4):
    problem_list = problem.split('|')
    eval_dict = {}
    for sub_problem in problem_list:
        params = Params()
        params.assign_problem(sub_problem, gpu=int(gpu))
        def input_fn(): return train_eval_input_fn(params, mode='eval')
        if 'ner' not in sub_problem and 'NER' not in sub_problem:
            eval_dict[sub_problem] = estimator.evaluate(input_fn=input_fn)
        else:
            pred = estimator.predict(input_fn=input_fn)
            pred_list = defaultdict(list)
            for p in pred:
                for pro in p:
                    pred_list[pro].append(p[pro])
            for pro in pred_list:
                if 'NER' in pro:
                    eval_dict[sub_problem] = ner_evaluate(
                        pro, pred_list[pro], params)
    return eval_dict
