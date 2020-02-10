import re

from ..utils import get_or_make_label_encoder
from ..special_tokens import (BOS_TOKEN,
                              EOS_TOKEN,
                              PREDICT)
from .preproc_decorator import preprocessing_fn


def parse_one(s):
    s = re.sub('\)', ') ', s)
    s = re.sub(' +', ' ', s).strip()
    pos = s.split(' ')
    buffer = []
    innermost = True
    full_pos = []
    seg = []
    ner = []
    pos_result = []
    ner_types = []
    pos_types = []
    text = []
    for p in pos:
        if '(' in p:
            innermost = True
            if 'NER' in p:
                ner_types.append(re.sub('NER', '', p[1:]))
                continue
            else:
                buffer.append(p)
        elif ')' in p:
            if buffer != []:
                pos_types = buffer[-1].replace('(', '')
                suffix = buffer.pop()[1:]
                if innermost:
                    assert len(p) > 1
                    word = p[:-1]
                    text.append(word)
                    innermost = False
                    p = p[-1]
                    if len(word) == 1:
                        seg_gt = ['O']
                    else:
                        seg_gt = ['B'] + ['M'] * (len(word) - 2) + ['E']
                    if ner_types != []:
                        ner_type = ner_types.pop()
                        ner_gt = [_ + ner_type for _ in seg_gt]
                    else:
                        ner_gt = ['O' for _ in seg_gt]
                    pos_gt = [_ + '-' + pos_types for _ in seg_gt]

                    seg.extend(seg_gt)
                    ner.extend(ner_gt)
                    pos_result.extend(pos_gt)
            p += suffix
        full_pos.append(p)
    text = list(''.join(text))
    return seg, ner, full_pos, text, pos_result


@preprocessing_fn
def ontonotes_ner(params, mode):
    if mode == 'train':
        with open('data/ontonote/train.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()
    else:
        with open('data/ontonote/test.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()

    _, target, _, inputs_list, _ = zip(*[parse_one(s) for s in raw_data])
    return inputs_list, target


@preprocessing_fn
def ontonotes_cws(params, mode):

    if mode == 'train':
        with open('data/ontonote/train.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()
    else:
        with open('data/ontonote/test.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()

    target, _, _, inputs_list, _ = zip(*[parse_one(s) for s in raw_data])
    return inputs_list, target


@preprocessing_fn
def ontonotes_chunk(params, mode):

    if mode == 'train':
        with open('data/ontonote/train.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()

        # some label not in train, weird
        with open('data/ontonote/test.fuse.parse', 'r', encoding='utf8') as f:
            test_raw_data = f.readlines()
        all_raw_data = raw_data + test_raw_data
        _, _, target, inputs_list, _ = zip(*[parse_one(s) for s in raw_data])
        _, _, all_target, _ = zip(*[parse_one(s) for s in all_raw_data])
        flat_target_list = [t for sublist in all_target for t in sublist]
        flat_target_list.extend([BOS_TOKEN, EOS_TOKEN])
    else:
        with open('data/ontonote/test.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()
        flat_target_list = None
        _, _, target, inputs_list, _ = zip(*[parse_one(s) for s in raw_data])

    label_encoder = get_or_make_label_encoder(
        params, 'ontonotes_chunk', mode, flat_target_list)
    return inputs_list, target


@preprocessing_fn
def ontonotes_pos(params, mode):

    if mode == 'train':
        with open('data/ontonote/train.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()

        # some label not in train, weird
        with open('data/ontonote/test.fuse.parse', 'r', encoding='utf8') as f:
            test_raw_data = f.readlines()
        all_raw_data = raw_data + test_raw_data
        _, _, _, inputs_list, target = zip(*[parse_one(s) for s in raw_data])
        _, _, _, _, all_target = zip(*[parse_one(s) for s in all_raw_data])
        flat_target_list = [t for sublist in all_target for t in sublist]
        flat_target_list.extend([BOS_TOKEN, EOS_TOKEN])
    else:
        with open('data/ontonote/test.fuse.parse', 'r', encoding='utf8') as f:
            raw_data = f.readlines()
        flat_target_list = None
        _, _, _, inputs_list, target = zip(*[parse_one(s) for s in raw_data])

    return inputs_list, target
