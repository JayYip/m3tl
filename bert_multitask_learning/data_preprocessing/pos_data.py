from sklearn.model_selection import train_test_split

from .ctb_data import read_ctb_pos
from .preproc_decorator import preprocessing_fn


@preprocessing_fn
def POS(params, mode):

    input_list, target_list = read_ctb_pos()

    if mode == 'train':
        input_list, _, target_list, _ = train_test_split(
            input_list, target_list, test_size=0.2, random_state=3721)
    else:
        _, input_list, _, target_list = train_test_split(
            input_list, target_list, test_size=0.2, random_state=3721)

    return input_list, target_list
