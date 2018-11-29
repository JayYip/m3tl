from glob import glob

from bert.tokenization import FullTokenizer

from ..utils import (get_or_make_label_encoder,
                     create_single_problem_generator)


def gold_horse_ent_type_process_fn(d):
    """golden horse ent type process fn
    Source: https://github.com/hltcoe/golden-ho rse

    Entity type:

        B, I, O: Begining \ In middle \ Outside of entity
        GPE: Country, City, District...
        LOC: Location, zoo, school...
        PER: Person
        ORG: Organiazation
        NAM: Entity
        NOM: More general, 女生, 男的...

        Example: 
            B-PER.NAM

    Only keep NAM here
    So after process:
        B-PER

    Arguments:
        ent_type {str} -- ent type from gold_horse data

    Returns:
        str -- processed enttype
    """
    ent_type = d.split('\t')[1].replace('\n', '')
    # keep nam only
    # ent_type = ent_type if 'NAM' in ent_type else 'O'
    # ent_type = ent_type.replace('.NAM', '')
    return ent_type


def chinese_literature_ent_type_process_fn(d):
    """Not match my need

    Arguments:
        d {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    ent_type = d.split(' ')[1].replace('\n', '')
    return ent_type


def read_ner_data(file_pattern='data/weiboNER*', proc_fn=None):
    """Read data from golden horse data


    Arguments:
        file_pattern {str} -- file patterns

    Returns:
        dict -- dict, key: 'train', 'eval', value: dict {'inputs', 'target'}
    """
    # if 'weiboNER' in file_pattern:
    #     proc_fn = gold_horse_ent_type_process_fn
    # elif 'Chinese-Literature' in file_pattern:
    #     proc_fn = chinese_literature_ent_type_process_fn

    result_dict = {
        'train': {
            'inputs': [],
            'target': []
        },
        'eval': {
            'inputs': [],
            'target': []
        }
    }
    file_list = glob(file_pattern)
    for file_path in file_list:
        raw_data = open(file_path, 'r', encoding='utf8').readlines()

        inputs_list = [[]]
        target_list = [[]]
        for d in raw_data:
            if d != '\n':
                # put first char to input
                inputs_list[-1].append(d[0])
                ent_type = proc_fn(d)
                target_list[-1].append(ent_type)
            else:
                inputs_list.append([])
                target_list.append([])

        # remove trailing empty str/list
        if not inputs_list[-1]:
            del inputs_list[-1]
        if not target_list[-1]:
            del target_list[-1]

        inputs_with_ent = []
        target_with_ent = []
        for inputs, target in zip(inputs_list, target_list):
            # if len(set(target)) > 1:
            inputs_with_ent.append(inputs)
            target_with_ent.append(target)

        if 'train' in file_path or 'dev' in file_path:
            result_dict['train']['inputs'] = inputs_with_ent
            result_dict['train']['target'] = target_with_ent
        else:
            result_dict['eval']['inputs'] = inputs_with_ent
            result_dict['eval']['target'] = target_with_ent
    return result_dict


def WeiboNER(params, mode):
    tokenizer = FullTokenizer(vocab_file=params.vocab_file)
    data = read_ner_data(file_pattern='data/weiboNER*',
                         proc_fn=gold_horse_ent_type_process_fn)
    if mode == 'train':
        data = data['train']
    else:
        data = data['eval']
    inputs_list = data['inputs']
    target_list = data['target']

    flat_label = [item for sublist in target_list for item in sublist]

    label_encoder = get_or_make_label_encoder(
        'WeiboNER', mode, flat_label)

    return create_single_problem_generator('WeiboNER',
                                           inputs_list,
                                           target_list,
                                           label_encoder,
                                           params,
                                           tokenizer, 0)


def WeiboFakeCLS(params, mode):
    """Just a test problem to test multiproblem support

    Arguments:
        params {Params} -- params
        mode {mode} -- mode
    """
    tokenizer = FullTokenizer(vocab_file=params.vocab_file)
    data = read_ner_data(file_pattern='data/weiboNER*',
                         proc_fn=gold_horse_ent_type_process_fn)
    if mode == 'train':
        data = data['train']
    else:
        data = data['eval']
    inputs_list = data['inputs']
    target_list = data['target']

    new_target_list = [1 if len(set(t)) > 1 else 0 for t in target_list]

    label_encoder = get_or_make_label_encoder(
        'WeiboFakeCLS', mode, new_target_list, 'O')

    return create_single_problem_generator('WeiboFakeCLS',
                                           inputs_list,
                                           new_target_list,
                                           label_encoder,
                                           params,
                                           tokenizer, 1)


def gold_horse_segment_process_fn(d):
    ent_type = d.split('\t')[0][-1]
    if ent_type not in ['0', '1', '2']:
        ent_type = '0'
    return ent_type


def WeiboSegment(params, mode):
    tokenizer = FullTokenizer(vocab_file=params.vocab_file)
    data = read_ner_data(file_pattern='data/weiboNER*',
                         proc_fn=gold_horse_segment_process_fn)
    if mode == 'train':
        data = data['train']
    else:
        data = data['eval']
    inputs_list = data['inputs']
    target_list = data['target']

    flat_label = [item for sublist in target_list for item in sublist]

    label_encoder = get_or_make_label_encoder(
        'WeiboSegment', mode, flat_label, '0')

    return create_single_problem_generator('WeiboSegment',
                                           inputs_list,
                                           target_list,
                                           label_encoder,
                                           params,
                                           tokenizer, 0)
