import unittest
import logging
import tensorflow as tf
import tempfile
import shutil

from ..params import Params
from ..model_fn import BertMultiTask
from ..estimator import Estimator
from ..ckpt_restore_hook import RestoreCheckpointHook
from ..input_fn import train_eval_input_fn, predict_input_fn


class TestProblems(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.params = Params()
        self.params.train_epoch = 1
        self.params.prefetch = 100
        self.params.shuffle_buffer = 100
        self.test_dir = tempfile.mkdtemp()

        self.dist_trategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus=int(1),
            cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
                'nccl', num_packs=int(1)))

        self.run_config = tf.estimator.RunConfig(
            train_distribute=self.dist_trategy,
            eval_distribute=self.dist_trategy,
            log_step_count_steps=self.params.log_every_n_steps)

    def test_seq2seq_tag(self):
        self.params.assign_problem(
            'weibo_fake_seq2seq_tag', gpu=1, base_dir=self.test_dir)

        model = BertMultiTask(self.params)
        model_fn = model.get_model_fn(False)
        estimator = Estimator(
            model_fn,
            model_dir=self.params.ckpt_dir,
            params=self.params,
            config=self.run_config)
        train_hook = RestoreCheckpointHook(self.params)

        def train_input_fn(): return train_eval_input_fn(self.params)
        estimator.train(
            train_input_fn,
            max_steps=self.params.train_steps,
            hooks=[train_hook])

        def input_fn(): return train_eval_input_fn(self.params, mode='eval')
        estimator.evaluate(input_fn=input_fn)

        p = estimator.predict(input_fn=input_fn)
        for _ in p:
            pass

    def tearDown(self):
        shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    unittest.main()
