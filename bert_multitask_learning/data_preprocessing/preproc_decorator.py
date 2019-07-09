import os

from ..utils import get_or_make_label_encoder, TRAIN, EVAL, PREDICT, cluster_alphnum
from ..create_generators import create_single_problem_generator, create_pretraining_generator
from ..tokenization import FullTokenizer


def preprocessing_fn(func):
    def wrapper(params, mode):
        problem = func.__name__
        pickle_file = os.path.join(
            params.tmp_file_dir, '{0}_{1}_data.pkl'.format(problem, mode))

        if params.problem_type[problem] != 'pretrain':
            if os.path.exists(pickle_file) and params.multiprocess:
                try:
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
                except TypeError:
                    pass

            tokenizer = FullTokenizer(
                vocab_file=params.vocab_file, do_lower_case=True)
            inputs_list, target_list = func(params, mode)

            label_encoder = get_or_make_label_encoder(
                params, problem=problem, mode=mode, label_list=target_list)
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
        else:
            tokenizer = FullTokenizer(
                vocab_file=params.vocab_file, do_lower_case=True)
            inputs_list = func(params, mode)

            params.num_classes['next_sentence'] = 2
            params.problem_type['next_sentence'] = 'cls'
            return create_pretraining_generator(func.__name__,
                                                inputs_list,
                                                None,
                                                None,
                                                params,
                                                tokenizer)

    return wrapper
