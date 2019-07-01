import os

from ..utils import get_or_make_label_encoder, TRAIN, EVAL, PREDICT, cluster_alphnum
from ..create_generators import create_single_problem_generator, create_pretraining_generator
from ..tokenization import FullTokenizer


def preprocessing_fn(func):
    def wrapper(params, mode):
        problem = func.__name__
        pickle_file = os.path.join(
            params.tmp_file_dir, '{0}_{1}_data.pkl'.format(problem, mode))
        if os.path.exists(pickle_file) and params.multiprocess:
            label_encoder = get_or_make_label_encoder(
                params, problem=problem, mode=mode)
            return create_single_problem_generator(
                func.__name__,
                None,
                None,
                None,
                params,
                None,
                mode)

        tokenizer = FullTokenizer(
            vocab_file=params.vocab_file, do_lower_case=True)
        inputs_list, target_list = func(params, mode)
        if isinstance(target_list[0], list):
            flat_target = [
                item for sublist in target_list for item in sublist]
        else:
            flat_target = target_list
        label_encoder = get_or_make_label_encoder(
            params, problem=problem, mode=mode, label_list=flat_target)
        if mode == PREDICT:
            return inputs_list, target_list, label_encoder
        return create_single_problem_generator(
            func.__name__,
            inputs_list,
            target_list,
            label_encoder,
            params,
            tokenizer,
            mode)
    return wrapper
