import copy
import itertools
import json
import logging
import os
import pickle
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import argparse
import pandas as pd

# from pyhanlp import HanLP
# from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

from src.estimator_wrapper import ChineseNER
from src.params import Params

ROOT_DIR = '.'
LOGGING_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"


def read_sougou_cell(file_path):
    raw_lines = open(file_path, 'r', encoding='utf8').readlines()
    return [l.split(' ')[2].strip('\n') for l in raw_lines]


class NERBaseWrapper():
    def __init__(self):
        return self

    @property
    def match_slot_file(self):
        """Slots to ONLY HARD MATCH.

        Returns:
            dict -- key: slot name, value: word table
        """
        slot_path_dict = {
            'PERSON': os.path.join(ROOT_DIR, 'data/person.txt'),
            'LOCATION': os.path.join(ROOT_DIR, 'data/location.txt')
        }
        for slot in slot_path_dict:
            if not os.path.exists(slot_path_dict[slot]):
                raise FileNotFoundError(
                    'File %s not found.' % slot_path_dict[slot])
            else:
                slot_path_dict[slot] = read_sougou_cell(slot_path_dict[slot])

        # xiuxiu star list
        star_list = [w.lower().replace('\n', '') for w in open(
            os.path.join(ROOT_DIR, 'data/xiuxiu_person.txt'),
            'r', encoding='utf8').readlines()]
        slot_path_dict['PERSON'] += star_list
        slot_path_dict['PERSON'] = list(set(slot_path_dict['PERSON']))

        return slot_path_dict

    @property
    def raw_match_slot_file(self):
        """Slot to recoginize with BOTH match and model.
        Match will be prioritized.

        Returns:
            dict -- key: slot name, value: word table
        """

        slot_path_dict = {
            'ORGANIZATION': os.path.join(ROOT_DIR, 'data/brand.txt')
        }
        for slot in slot_path_dict:
            if not os.path.exists(slot_path_dict[slot]):
                raise FileNotFoundError(
                    'File %s not found.' % slot_path_dict[slot])
            else:
                slot_path_dict[slot] = set([w.lower().replace('\n', '') for w in open(
                    slot_path_dict[slot], 'r', encoding='utf8').readlines()])

        return slot_path_dict

    def word_match_with_table(self, word):
        """Match word with word table

        Arguments:
            word {str} -- Word to match

        Returns:
            str or None -- entity type or None if not matched
        """

        if self.match_table is None:
            self.match_table = dict()

            for slot in self.match_slot_file:
                self.match_table[slot] = set(self.match_slot_file[slot])

        for ent_type in self.match_table:
            if word in self.match_table[ent_type]:
                return ent_type

        return None

    def eng_brand_extract(self, text, slot_type):

        wordset = self.raw_match_slot_file[slot_type]

        # 删除话题tag、连续中文段替换为<split>、连续的空格替换为单空格、英文全转小写、按<split>分词

        # text = re.sub('[\u4e00-\u9fa5\n\r]|[# +]', '<split>',
        #               text.lower()).split('<split>')
        text_list = re.findall(r'[A-Za-z \']+', text)
        lower_text_list = [t.lower() for t in text_list]
        eng_keys = [t.strip()
                    for t in set(lower_text_list) & wordset]  # 存在英文品牌

        return_ent = []
        for k in eng_keys:
            return_ent.append(text_list[lower_text_list.index(k)])

        return return_ent

    @property
    def exclude_slot_file(self):
        slot_path_dict = {
            'PERSON': os.path.join(ROOT_DIR, 'data/exclude_person.txt'),
            'CITY': os.path.join(ROOT_DIR, 'data/exclude_city.txt')
        }
        for slot in slot_path_dict:
            if not os.path.exists(slot_path_dict[slot]):
                raise FileNotFoundError(
                    'File %s not found.' % slot_path_dict[slot])
            else:
                slot_path_dict[slot] = set([w.lower().replace('\n', '') for w in open(
                    slot_path_dict[slot], 'r', encoding='utf8').readlines()])
        return slot_path_dict

    def exclude_with_table(self, ent_dict):
        """exclude with word table

        Arguments:
            word {str} -- Word to match

        Returns:
            str or None -- entity type or None if not matched
        """
        for slot_type in self.exclude_slot_file:
            if slot_type in ent_dict:
                ent_dict[slot_type] = list(set(ent_dict[slot_type]).difference(
                    self.exclude_slot_file[slot_type]))
        return ent_dict

    def empty_dict(self, d):
        return len(d) == 0

    def make_model_match_ent_type(self, ent_type_set):
        """Get the model ner slots and update the matching slots

        Arguments:
            ent_type_set {set} -- entity type to recognize

        Returns:
            set -- model ner type set
        """

        model_ner_type_set = ent_type_set.difference(
            self.match_slot_file.keys())
        for slot in self.match_slot_file:
            if slot not in ent_type_set:
                del self.match_slot_file[slot]
        return model_ner_type_set

    def argument_check(self, inputs, ent_type_set):

        if not isinstance(ent_type_set, set):
            ent_type_set = set(ent_type_set)

        return ent_type_set

    def create_ent_dict(self, inputs, ent_type_set):
        ent_type_set = self.argument_check(inputs, ent_type_set)

        ent_dict = defaultdict(list)

        # match with brand
        brand_list = list(self.eng_brand_extract(
            inputs, 'ORGANIZATION'))
        if brand_list:
            ent_dict['ORGANIZATION'] = brand_list

        clean_text, ent_dict = self.inputs_at_handling(inputs, ent_dict)

        return inputs, clean_text, ent_dict, ent_type_set

    def inputs_at_handling(self, inputs, ent_dict):
        """This function will handle @XXX in feed and return
        @XXX removed text and updated ent_dict. XXX will be matched
        with person table.

        Arguments:
            inputs {str} -- input text
            ent_dict {dict} -- entity dict
        """
        if '@' not in inputs:
            return inputs, ent_dict
        else:
            clean_text = copy.copy(inputs)

            at_index = []
            location = -1
            # Loop while true.
            while True:
                # Advance location by 1.
                location = inputs.find("@", location + 1)
                # Break if not found.
                if location == -1:
                    break
                at_index.append(location)

            for ind in at_index:
                rest_text = re.split(' |@|#', inputs[ind:])
                rest_text = [t for t in rest_text if t]
                #rest_text = inputs[ind:].split(' ')

                if len(rest_text) > 1:
                    at_person = rest_text[0].replace('@', '')
                elif len(rest_text) > 0:
                    if len(rest_text[0]) < 10:
                        at_person = rest_text[0].replace('@', '')
                    else:
                        continue
                else:
                    continue

                ent_type = self.word_match_with_table(at_person)
                if ent_type == 'PERSON':
                    ent_dict['PERSON'].append(at_person)

                clean_text = clean_text.replace('@%s' % at_person, '')

            return clean_text, ent_dict

    def order_by_occurence(self, inputs, ent_dict):
        occurence_list = []

        for ent_type in ent_dict:
            for ent in ent_dict[ent_type]:
                occurence_list.append((ent_type, ent, inputs.count(ent)))

        ordered_list = sorted(
            occurence_list, key=lambda tup: tup[2], reverse=True)
        ordered_list = [t[:-1] for t in ordered_list]

        return ordered_list

    def ner(self,
            inputs,
            extract_ent=True,
            ent_type_set=('PERSON', 'LOCATION', 'ORGANIZATION', 'COUNTRY', 'CITY', 'STATE_OR_PROVINCE')):
        raise NotImplementedError


class StanfordNLPWrapper(NERBaseWrapper):
    def __init__(self, model_dir: str, lang='zh'):
        self.model_dir = model_dir
        self.lang = lang
        self.model = StanfordCoreNLP(model_dir, lang=lang, port=9000)

        self.match_table = None

    def ner(self,
            inputs,
            extract_ent=True,
            ent_type_set=('PERSON', 'LOCATION', 'ORGANIZATION', 'COUNTRY', 'CITY', 'STATE_OR_PROVINCE')):
        """function to predict ner from inputs, if extract_ent is true, will return a
        dict of list with entities. Else, return processed string with entities masked.

        Arguments:
            inputs {str} -- input string

        Keyword Arguments:
            extract_ent {bool} -- Whether to extrac ent or mask entity (default: {True})
            ent_type {tuple} -- entity type to extract or mask (default: {('PERSON', 'ORGANIZATION', 'COUNTRY', 'CITY')})

        Raises:
            TypeError -- inputs is not string

        Returns:
            str or dict -- entity dict or masked string
        """
        if not isinstance(inputs, str):
            return []

        raw_inputs, inputs, ent_dict, ent_type_set = self.create_ent_dict(
            inputs, ent_type_set)

        buff = []

        try:
            ner_results = self.model.ner(inputs)
        except:
            if extract_ent:
                return []
            else:
                return ''

        model_ner_type_set = self.make_model_match_ent_type(ent_type_set)
        if extract_ent:
            for _, (ent, ent_type) in enumerate(ner_results):
                match_type = self.word_match_with_table(ent)
                if match_type is not None:
                    ent_dict[match_type].append(ent)
                elif ent_type in model_ner_type_set:
                    ent_dict[ent_type].append(ent)

            ent_dict = self.exclude_with_table(ent_dict)
            ent_list = self.order_by_occurence(raw_inputs, ent_dict)
            return ent_list

        else:
            for _, (ent, ent_type) in enumerate(ner_results):
                match_type = self.word_match_with_table(ent)
                if match_type is not None:
                    replace_str = '@@%s@@' % match_type
                elif ent_type in model_ner_type_set:
                    replace_str = '@@%s@@' % ent_type
                else:
                    replace_str = ent

                buff.append(replace_str)

            processed = ''.join(buff)

            return processed


class HanlpWrapper(NERBaseWrapper):
    def __init__(self):

        self.model = HanLP.newSegment().enableAllNamedEntityRecognize(True)

        self.match_table = None

    @property
    def nature2ent(self):
        return {
            'ns': ['LOCATION', 'COUNTRY', 'CITY', 'STATE_OR_PROVINCE'],
            'nr': 'PERSON',
            'nt': 'ORGANIZATION',
            'ni': 'ORGANIZATION'
        }

    @property
    def ent2nature(self):
        return {
            'LOCATION': 'ns',
            'COUNTRY': 'ns',
            'CITY': 'ns',
            'STATE_OR_PROVINCE': 'ns',
            'PERSON': 'nr',
            'ORGANIZATION': ['nt', 'ni'],
        }

    def get_root_nature(self, nature):
        for rn in self.nature2ent:
            if rn in nature:
                return rn

        return nature

    def ner(self,
            inputs,
            extract_ent=True,
            ent_type_set=('PERSON', 'LOCATION', 'ORGANIZATION', 'COUNTRY', 'CITY', 'STATE_OR_PROVINCE')):
        """function to predict ner from inputs, if extract_ent is true, will return a
        dict of list with entities. Else, return processed string with entities masked.

        Arguments:
            inputs {str} -- input string

        Keyword Arguments:
            extract_ent {bool} -- Whether to extrac ent or mask entity (default: {True})
            ent_type {tuple} -- entity type to extract or mask (default: {('PERSON', 'ORGANIZATION', 'COUNTRY', 'CITY')})

        Raises:
            TypeError -- inputs is not string

        Returns:
            str or dict -- entity dict or masked string
        """
        if not isinstance(inputs, str):
            return []

        raw_inputs, inputs, ent_dict, ent_type_set = self.create_ent_dict(
            inputs, ent_type_set)

        try:
            ner_results = self.model.seg(inputs)
        except:
            if extract_ent:
                return []
            else:
                return ''

        model_ner_type_set = self.make_model_match_ent_type(ent_type_set)
        if extract_ent:
            for ent_ent_type in ner_results:
                # try:
                #     ent, ent_type = ent_ent_type.toString().split('/')
                # except ValueError:
                #     print(ent_ent_type)
                #     raise ValueError
                split_list = ent_ent_type.toString().split('/')
                if len(split_list) == 2:
                    ent, ent_type = split_list
                else:
                    continue
                match_type = self.word_match_with_table(ent)
                if match_type is not None:
                    ent_dict[match_type].append(ent)

                ent_type = self.get_root_nature(ent_type)
                if ent_type in self.nature2ent:
                    ent_type = self.nature2ent[ent_type]

                    if isinstance(ent_type, str) and ent_type in model_ner_type_set:
                        ent_dict[ent_type].append(ent)

                    elif isinstance(ent_type, list):
                        for et in ent_type:
                            if et in model_ner_type_set:
                                ent_dict[et].append(ent)

            ent_dict = self.exclude_with_table(ent_dict)
            ent_list = self.order_by_occurence(raw_inputs, ent_dict)
            return ent_list

        else:
            raise NotImplementedError(
                'HanlpWrapper not implement mask comments.')


class NER(NERBaseWrapper):
    def __init__(self, ensemble_type='intersection'):
        self.ensemble_type = ensemble_type
        # self.model_dict = {
        #     'StanfordCoreNLP': StanfordNLPWrapper('http://localhost', lang='zh'),
        #     'Hanlp': HanlpWrapper()}
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            format=LOGGING_FORMAT, level=logging.INFO)
        self.logger.setLevel(logging.WARNING)

        self.match_table = None

    def models_intersect_ensemble(self, ent_type_set, result_dict):

        result_list = []
        for model_name in result_dict:
            result_list.append(result_dict[model_name])

        final_result = sorted(
            set(result_list[0]).intersection(set(*result_list[1:])),
            key=result_list[0].index)

        return final_result

    def models_union_ensemble(self, ent_type_set, result_dict, high_priority='StanfordCoreNLP'):
        result_list = []
        for model_name in result_dict:
            result_list.append(result_dict[model_name])

        final_result =\
            set(result_list[0]).union(set(*result_list[1:]))

        return final_result

    def ner(self,
            inputs,
            extract_ent=True,
            ent_type_set=('PERSON', 'LOCATION', 'ORGANIZATION',
                          'COUNTRY', 'CITY', 'STATE_OR_PROVINCE'),
            ensemble_type=None):
        if not isinstance(inputs, str):
            return []

        if ensemble_type is None:
            ensemble_type = self.ensemble_type

        result_dict = defaultdict(str)
        for model_name in self.model_dict:
            try:
                result_dict[model_name] = self.model_dict[model_name].ner(
                    inputs, extract_ent, ent_type_set)
            except NotImplementedError:
                self.logger.warning('%s not impletmented. Skip.' % model_name)
                continue

        if extract_ent:
            if ensemble_type == 'intersection':
                result = self.models_intersect_ensemble(
                    ent_type_set, result_dict)
            else:
                result = self.models_union_ensemble(ent_type_set, result_dict)

            return result
        else:
            return result_dict[list(result_dict.keys())[0]]


def _multiprocess_ner_results(data_list: list):
    result_dict = defaultdict(list)
    # stanford_ner_model = StanfordNLPWrapper('http://localhost')
    # hanlp_ner_model = HanlpWrapper()
    ner_model = NER(ensemble_type='intersection')

    # Multithread call stanford
    if not os.path.exists('tmp/stf_ner.pkl'):
        with Pool(cpu_count()) as p:
            stanford_inputs_ner_list = list(
                tqdm(p.imap(stanford_ner_model.ner, data_list),
                     total=len(data_list), desc='Stanford NER'))
        # stanford_inputs_ner_list = [stanford_ner_model.ner(
        #     t) for t in tqdm(data_list, total=len(data_list), desc='Hanlp NER')]

        pickle.dump(stanford_inputs_ner_list, open('tmp/stf_ner.pkl', 'wb'))
    else:
        stanford_inputs_ner_list = pickle.load(open('tmp/stf_ner.pkl', 'rb'))

    if not os.path.exists('tmp/han_ner.pkl'):
        hanlp_inputs_ner_list = [hanlp_ner_model.ner(
            t) for t in tqdm(data_list, total=len(data_list), desc='Hanlp NER')]
        pickle.dump(hanlp_inputs_ner_list, open('tmp/han_ner.pkl', 'wb'))
    else:
        hanlp_inputs_ner_list = pickle.load(open('tmp/han_ner.pkl', 'rb'))

    result_dict['Stanford'] = stanford_inputs_ner_list
    result_dict['Hanlp'] = hanlp_inputs_ner_list

    ent_type_set = ('PERSON', 'LOCATION', 'ORGANIZATION',
                    'COUNTRY', 'CITY', 'STATE_OR_PROVINCE')

    for stanford, hanlp in zip(stanford_inputs_ner_list, hanlp_inputs_ner_list):
        tmp_dict = {}
        tmp_dict['StanfordCoreNLP'] = stanford
        tmp_dict['Hanlp'] = hanlp
        result_dict['Intersection'].append(
            ner_model.models_intersect_ensemble(ent_type_set, tmp_dict))
        result_dict['Union'].append(
            ner_model.models_union_ensemble(ent_type_set, tmp_dict))

    return result_dict


def get_smallest_label(label_str):
    """Get the smallest label from label json str

    Example:
        label_str = [{
            '一级标签': 'a',
            '二级标签': 'b'
        },
        {
            '一级标签': 'c'
        }]

        print(get_smallest_label(label_str))

    Expected results:
        ['b', 'c']

    Arguments:
        label_str {str} -- json str

    Returns:
        list -- list of tags
    """
    if isinstance(label_str, str):
        js = json.loads(label_str)
    else:
        js = label_str
    label_list = []

    for label_set in js:
        if 'tagLevel1' not in label_set:
            label_order = ['三级标签', '二级标签', '一级标签']
        else:
            label_order = ['tagLevel3', 'tagLevel2', 'tagLevel1']

        for label in label_order:
            if label in label_set:
                label_list.append(label_set[label])
                break

    return label_list


def read_hot_feed(file_path, keep_feed_text=False, exclude_with_no_comments=False):
    """Read hot feed file and do basic filtering

    - With tags
    - With comments

    Arguments:
        file_path {str} -- file path to hot feed file

    Keyword Arguments:
        keep_feed_text {bool} -- Whether to keep feed text in return df (default: {False})

    Returns:
        dataframe -- dataframe of hot feed data
    """

    hot_feed = pd.read_csv(file_path, encoding='gbk')

    hot_feed = hot_feed[hot_feed['内容标签'].notnull()]

    hot_feed['labels'] = hot_feed['内容标签'].apply(get_smallest_label)

    if exclude_with_no_comments:
        #  with comments
        hot_feed = hot_feed[hot_feed['评论数'] > 0]

    hot_feed['feed_id'] = hot_feed.ID.apply(lambda x: x.strip('\t'))

    if keep_feed_text:
        hot_feed['feed'] = hot_feed['描述']
        return hot_feed.loc[:, ['feed_id', 'feed', 'labels']]
    else:
        return hot_feed.loc[:, ['feed_id', 'labels']]


def order_dedup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def main(args):
    if args.debug:
        hot_feed = read_hot_feed(
            'tmp/hotfeed-2018-10-31-15224021.csv', keep_feed_text=True)
        hot_feed = hot_feed[:100]
    else:
        hot_feed = read_hot_feed(
            'tmp/hotfeed-2018-10-31-15224021.csv', keep_feed_text=True)

    logger = logging.getLogger()
    logger.disabled = True

    ner_result_dict = _multiprocess_ner_results(hot_feed.feed.tolist())
    hot_feed.reset_index(drop=True, inplace=True)

    file_path = 'tmp/ner_comparison.csv' if args.output is None\
        else args.output

    # set up bert estimator
    params = Params()
    bert_ner = ChineseNER(
        params, model_dir='tmp/CTBCWS_CTBPOS_CWS_NER_ckpt/', gpu=2)
    hot_feed_text = [t if isinstance(
        t, str) else '' for t in hot_feed.feed.tolist()]
    # assert len(hot_feed_text) == len(ner_result_dict['Intersection'])
    bert_ner_result = bert_ner.ner(hot_feed_text)
    bert_ner_result = [order_dedup(n) for n in bert_ner_result]

    with open(file_path, 'w', encoding='utf8') as f:
        f.write('feed_id,feed,Stanford,BertCRF\n')
        for row_ind, row in tqdm(enumerate(hot_feed.itertuples(index=False)),
                                 total=hot_feed.shape[0]):

            if not isinstance(row.feed, str):
                continue
            response = {}
            for model in ['Stanford', 'Intersection']:
                input_ner = order_dedup(ner_result_dict[model][row_ind])

                response[model] = input_ner

            response['BertCRF'] = bert_ner_result[row_ind]

            write_feed = ' '.join(row.feed.splitlines())

            f.write('"%s","%s","%s","%s","%s","%s"\n' %
                    (row.feed_id,
                     write_feed,
                     row.labels,
                     response['Stanford'],
                     response['Intersection'],
                     response['BertCRF']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument('--mode', dest='mode', default='eval_hotfeed',
                        help='Eval mode')
    parser.add_argument('--debug', dest='debug', default=False, help='Debug')
    parser.add_argument('--output', dest='output',
                        default=None, help='Output file path')
    args = parser.parse_args()
    mode = args.mode

    main(args)
