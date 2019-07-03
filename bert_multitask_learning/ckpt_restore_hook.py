import tensorflow as tf

from . import modeling


class RestoreCheckpointHook(tf.train.SessionRunHook):
    def __init__(self,
                 params
                 ):
        tf.logging.info("Create RestoreCheckpointHook.")

        self.params = params
        self.checkpoint_path = params.init_checkpoint

    def begin(self):
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(
            tvars, self.params.init_checkpoint)
        tf.train.init_from_checkpoint(
            self.params.init_checkpoint, assignment_map)

        self.saver = tf.train.Saver(tvars)

    def after_create_session(self, session, coord):

        pass

    def before_run(self, run_context):
        return None

    def after_run(self, run_context, run_values):
        pass

    def end(self, session):
        pass
