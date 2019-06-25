from ..utils import get_or_make_label_encoder, TRAIN, EVAL, PREDICT, cluster_alphnum
from ..create_generators import create_single_problem_generator, create_pretraining_generator
from ..tokenization import FullTokenizer


def proprocessing_fn(func):
    def wrapper(params, mode):
        tokenizer = FullTokenizer(
            vocab_file=params.vocab_file, do_lower_case=True)
        inputs_list, target_list = func(params, mode)
        if isinstance(target_list[0], list):
            flat_target = [
                item for sublist in target_list for item in sublist]
        else:
            flat_target = target_list
        label_encoder = get_or_make_label_encoder(
            params, problem=func.__name__, mode=mode, label_list=flat_target)
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
