import tensorflow as tf

from . import modeling


class RestoreCheckpointHook(tf.estimator.SessionRunHook):
    """Deprecated
    """

    def __init__(self,
                 params
                 ):
        tf.compat.v1.logging.info("Create RestoreCheckpointHook.")

        self.params = params
        self.checkpoint_path = params.init_checkpoint

    def begin(self):
        tvars = tf.compat.v1.trainable_variables()
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(
            tvars, self.params.init_checkpoint)
        tf.compat.v1.train.init_from_checkpoint(
            self.params.init_checkpoint, assignment_map)

        self.saver = tf.compat.v1.train.Saver(tvars)

    def after_create_session(self, session, coord):

        pass

    def before_run(self, run_context):
        return None

    def after_run(self, run_context, run_values):
        pass

    def end(self, session):
        pass
