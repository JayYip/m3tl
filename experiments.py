import time
from collections import defaultdict
import os
import pickle

import tensorflow as tf

from src.input_fn import train_eval_input_fn, predict_input_fn
from src.metrics import ner_evaluate
from src.model_fn import BertMultiTask
from src.params import Params
from src.utils import create_path
from src.estimator import Estimator
from src.ckpt_restore_hook import RestoreCheckpointHook


EXPERIMENTS_LIST = [
    {'problems': ['pkucws',
                  'cityucws', 'msrcws', 'WeiboNER', 'bosonner', 'msraner',
                  'CTBCWS', 'CTBPOS', 'ascws'],

     'additional_params': {},
     'name': 'baseline'},
    {
        'name': 'mix_data_baseline',
        'problems': ['CWS', 'NER', 'POS'],

        'additional_params': {}
    },
    {'name': 'multitask_baseline',
        'problems': ['CWS|NER|POS'],
        'additional_params': {}
     }
]

# EXPERIMENTS_LIST = [
#     {'problems': ['WeiboNER', 'WeiboSegment'],

#      'additional_params': {},
#      'name': 'baseline'},

# ]


def train_problem(params, problem, gpu=4, base='baseline'):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    base = os.path.join('tmp', base)
    params.assign_problem(problem, gpu=int(gpu), base_dir=base)

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


def eval_single_problem(params, problem, estimator, gpu=4, base='baseline'):

    params.assign_problem(problem, gpu=int(gpu), base_dir=base)
    eval_dict = {}

    def input_fn(): return train_eval_input_fn(params, mode='eval')
    if 'ner' not in problem and 'NER' not in problem:
        eval_dict.update(estimator.evaluate(input_fn=input_fn))
    else:
        pred = estimator.predict(input_fn=input_fn)
        pred_list = defaultdict(list)
        for p in pred:
            for pro in p:
                pred_list[pro].append(p[pro])
        for pro in pred_list:
            if 'NER' in pro:
                raw_ner_eval = ner_evaluate(
                    pro, pred_list[pro], params)
                rename_dict = {}
                rename_dict['%s_Accuracy' % pro] = raw_ner_eval['Acc']
                rename_dict['%s_F1 Score' % pro] = raw_ner_eval['F1']
                rename_dict['%s_Precision' % pro] = raw_ner_eval['Precision']
                rename_dict['%s_Recall' % pro] = raw_ner_eval['Recall']
                eval_dict.update(rename_dict)
    return eval_dict


def eval_problem(params, raw_problem, estiamtor, gpu=4, base='baseline'):
    eval_problem_list = []
    base = os.path.join('tmp', base)
    for sub_problem in raw_problem.split('|'):
        if sub_problem == 'CWS':
            eval_problem_list += ['ascws', 'msrcws', 'pkucws',
                                  'cityucws', 'CTBCWS']
        elif sub_problem == 'NER':
            eval_problem_list += ['WeiboNER', 'bosonner', 'msraner']
        elif sub_problem == 'POS':
            eval_problem_list += ['CTBPOS']
        else:
            eval_problem_list.append(sub_problem)

    final_eval_dict = {}
    for p in eval_problem_list:
        final_eval_dict.update(eval_single_problem(
            params, p, estiamtor, gpu=gpu, base=base))
    return final_eval_dict


def main():
    params = Params()

    if os.path.exists('tmp/results.pkl'):
        with open('tmp/results.pkl', 'rb') as f:
            result_dict = pickle.load(f)
    else:
        result_dict = defaultdict(dict)
    for experiment_set in EXPERIMENTS_LIST:
        print('Running Problem set %s' % experiment_set['name'])
        if experiment_set['additional_params']:
            for k, v in experiment_set['additional_params'].items():
                setattr(params, k, v)
        for problem in experiment_set['problems']:
            if '%s_Accuracy' % problem not in result_dict[experiment_set['name']]:
                estiamtor = train_problem(
                    params, problem, 3, experiment_set['name'])
                eval_dict = eval_problem(
                    params, problem, estiamtor, 3, base=experiment_set['name'])
                result_dict[experiment_set['name']].update(eval_dict)
                print(result_dict)
                pickle.dump(result_dict, open('tmp/results.pkl', 'wb'))

    print(result_dict)

    pickle.dump(result_dict, open('tmp/results.pkl', 'wb'))


if __name__ == '__main__':
    main()
