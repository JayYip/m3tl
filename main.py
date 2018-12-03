import time
from collections import defaultdict
import os

import tensorflow as tf

from src.input_fn import train_eval_input_fn
from src.metrics import ner_evaluate
from src.model_fn import BertWrapper
from src.params import Params
from src.utils import create_path
from src.estimator import Estimator
from src.ckpt_restore_hook import RestoreCheckpointHook

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("problem", "WeiboNER",
                    "Problems to run, for multiproblem, use & to seperate, e.g. WeiboNER&WeiboSegment")

flags.DEFINE_string("schedule", "train",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer("gpu", 2,
                     "number of gpu to use")

PROBLEMS_LIST = [
    'WeiboNER',
    'WeiboSegment',
    'WeiboFakeCLS'
]


def main(_):

    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    params = Params()
    params.assign_problem(FLAGS.problem)

    create_path(params.ckpt_dir)

    tf.logging.info('Checkpoint dir: %s' % params.ckpt_dir)
    time.sleep(3)

    model = BertWrapper(params=params)
    model_fn = model.get_model_fn(warm_start=False)

    dist_trategy = tf.contrib.distribute.MirroredStrategy(
        num_gpus=int(FLAGS.gpu),
        cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
            'nccl', num_packs=int(FLAGS.gpu)))

    run_config = tf.estimator.RunConfig(
        train_distribute=dist_trategy,
        log_step_count_steps=params.log_every_n_steps)

    # ws = make_warm_start_setting(params)

    estimator = Estimator(
        model_fn,
        model_dir=params.ckpt_dir,
        params=params,
        config=run_config)

    if FLAGS.schedule == 'train':
        train_hook = RestoreCheckpointHook(params)

        def train_input_fn(): return train_eval_input_fn(params)
        estimator.train(
            train_input_fn, max_steps=params.train_steps, hooks=[train_hook])

        def input_fn(): return train_eval_input_fn(params, mode='eval')
        estimator.evaluate(input_fn=input_fn)
        pred = estimator.predict(input_fn=input_fn)

        pred_list = defaultdict(list)
        for p in pred:
            for problem in p:
                pred_list[problem].append(p[problem])
        for problem in pred_list:
            if 'NER' in problem:
                ner_evaluate(problem, pred_list[problem], params)

    elif FLAGS.schedule == 'eval':

        for _ in range(100):
            def input_fn(): return train_eval_input_fn(params, mode='eval')
            estimator.evaluate(input_fn=input_fn)
            pred = estimator.predict(input_fn=input_fn)

            pred_list = defaultdict(list)
            for p in pred:
                for problem in p:
                    pred_list[problem].append(p[problem])
            for problem in pred_list:
                if 'NER' in problem:
                    ner_evaluate(problem, pred_list[problem], params)

            time.sleep(400)

    elif FLAGS.schedule == 'predict':
        def input_fn(): return train_eval_input_fn(params, mode='eval')
        pred = estimator.predict(input_fn=input_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
