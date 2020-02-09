
from sklearn.model_selection import train_test_split

from .preproc_decorator import preprocessing_fn


@preprocessing_fn
def emotion_analysis(params, mode):
    with open('data/emotion_analysis/mer.negative.courpus_and_tag2.txt') as f:
        neg_data = [list(t.replace(' ', '')) for t in f.readlines()]

    with open('data/emotion_analysis/mer.positive.courpus_and_tag2.txt') as f:
        pos_data = [list(t.replace(' ', '')) for t in f.readlines()]

    neg_label = ['1' for _ in neg_data]
    pos_label = ['0' for _ in pos_data]

    all_inputs = neg_data + pos_data
    all_label = neg_label + pos_label

    train_input, eval_input, train_target, eval_target = train_test_split(
        all_inputs, all_label,  test_size=0.2, random_state=1024)

    if mode == 'train':
        inputs_list = train_input
        target_list = train_target
    else:
        inputs_list = eval_input
        target_list = eval_target

    return inputs_list, target_list
