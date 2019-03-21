import time
import os

import tensorflow as tf

from src.input_fn import train_eval_input_fn, predict_input_fn
from src.model_fn import BertMultiTask
from src.params import Params
from src.estimator import Estimator
from src.ckpt_restore_hook import RestoreCheckpointHook
from src import metrics
from src.utils import TRAIN, EVAL, PREDICT

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("problem", "weibo_ner",
                    "Problems to run, for multiproblem, use & to seperate, e.g. weibo_ner&weibo_cws")

flags.DEFINE_string("schedule", "train",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer("gpu", 2,
                     "number of gpu to use")

flags.DEFINE_string("model_dir", "",
                    "Model dir. If not specified, will use problem_name + _ckpt")
flags.DEFINE_string("eval_scheme", "ner",
                    "Evaluation Scheme, Example: ner/cws")


def main(_):

    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    if FLAGS.model_dir:
        base_dir, dir_name = os.path.split(FLAGS.model_dir)
    else:
        base_dir, dir_name = None, None

    params = Params()
    params.assign_problem(FLAGS.problem, gpu=int(FLAGS.gpu),
                          base_dir=base_dir, dir_name=dir_name)

    tf.logging.info('Checkpoint dir: %s' % params.ckpt_dir)
    time.sleep(3)

    model = BertMultiTask(params=params)
    model_fn = model.get_model_fn(warm_start=False)

    dist_trategy = tf.contrib.distribute.MirroredStrategy(
        num_gpus=int(FLAGS.gpu),
        cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
            'nccl', num_packs=int(FLAGS.gpu)))

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

    if FLAGS.schedule == 'train':
        train_hook = RestoreCheckpointHook(params)

        def train_input_fn(): return train_eval_input_fn(params)
        estimator.train(
            train_input_fn, max_steps=params.train_steps, hooks=[train_hook])

        def input_fn(): return train_eval_input_fn(params, mode='eval')
        estimator.evaluate(input_fn=input_fn)

    elif FLAGS.schedule == 'eval':

        params.from_json()
        evaluate_func = getattr(metrics, FLAGS.eval_scheme+'_evaluate')
        print(evaluate_func(FLAGS.problem, estimator, params))

    elif FLAGS.schedule == 'predict':
        def input_fn(): return predict_input_fn(
            ['''兰心餐厅\n作为一个无辣不欢的妹子，对上海菜的偏清淡偏甜真的是各种吃不惯。
            每次出门和闺蜜越饭局都是避开本帮菜。后来听很多朋友说上海有几家特别正宗味道做
            的很好的餐厅于是这周末和闺蜜们准备一起去尝一尝正宗的本帮菜。\n进贤路是我在上
            海比较喜欢的一条街啦，这家餐厅就开在这条路上。已经开了三十多年的老餐厅了，地
            方很小，就五六张桌子。但是翻桌率比较快。二楼之前的居民间也改成了餐厅，但是在
            上海的名气却非常大。烧的就是家常菜，普通到和家里烧的一样，生意非常好，外面排
            队的比里面吃的人还要多。'''], params, mode=PREDICT)
        pred = estimator.predict(input_fn=input_fn)
        for p in pred:
            print(p)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
