

import tensorflow as tf


from .params import Params
from .utils import create_generator


def train_eval_input_fn(config: Params, mode='train', epoch=None):

    def gen():
        if mode == 'train':
            epoch = config.train_epoch
        else:
            epoch = 1

        g = create_generator(params=config, mode=mode, epoch=epoch)
        for example in g:
            yield example

    output_type = {
        'input_ids': tf.int32,
        'input_mask': tf.int32,
        'segment_ids': tf.int32
    }
    output_shapes = {
        'input_ids': [config.max_seq_len],
        'input_mask': [config.max_seq_len],
        'segment_ids': [config.max_seq_len]
    }
    for problem, problem_type in config.problem_type.items():
        output_type.update({'%s_label_ids' % problem: tf.int32})
        if problem_type in ['seq_tag']:
            output_shapes.update(
                {'%s_label_ids' % problem: [config.max_seq_len]})
        elif problem_type in ['cls']:
            output_shapes.update({'%s_label_ids' % problem: []})

    dataset = tf.data.Dataset.from_generator(
        gen, output_types=output_type, output_shapes=output_shapes)
    if mode == 'train':
        dataset = dataset.shuffle(1000)

    dataset = dataset.prefetch(1000)
    if mode == 'train':
        dataset = dataset.batch(config.batch_size)
    else:
        dataset = dataset.batch(config.batch_size*2)
    return dataset


def no_dataset_input_fn(config: Params, mode='train', epoch=None):
    """This function is for evaluation only

    Arguments:
        config {Params} -- Param

    Keyword Arguments:
        mode {str} -- Mode (default: {'train'})
        epoch {int} -- epoch (default: {None})
    """

    g = create_generator(params=config, mode=mode, epoch=1)
    for example in g:
        yield example
