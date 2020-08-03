import numpy as np
from nltk.translate import bleu_score

from .input_fn import predict_input_fn
from .special_tokens import PREDICT


def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    return accuracy, precision, recall, f_measure


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + \
        input_string[0:target_position]
    return output_string


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(
                begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(
                single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(
                    begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(
                    begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def ner_evaluate(problem, estimator, params):
    text, label_data, label_encoder = params.read_data_fn[problem](
        params, PREDICT)

    t_l_tuple_list = list(zip(text, label_data))
    t_l_tuple_list = sorted(t_l_tuple_list, key=lambda t: len(t[0]))
    text, label_data = zip(*t_l_tuple_list)

    def pred_input_fn(): return predict_input_fn(text, params, mode=PREDICT)

    pred_list = estimator.predict(pred_input_fn)

    decode_pred_list = []
    decode_label_list = []

    scope_name = params.share_top[problem]

    for p, label, t in zip(pred_list, label_data, text):
        if not t:
            continue
        true_seq_length = len(t) - 1
        pred_prob = p[scope_name]

        pred_prob = pred_prob[1:true_seq_length]

        # crf returns tags
        predict = pred_prob
        label = label[:len(predict)]
        assert len(pred_prob) == len(label), print(len(pred_prob), len(label))

        if not params.crf:
            predict = np.argmax(predict, axis=-1)
        decode_pred = label_encoder.inverse_transform(predict)

        decode_pred_list.append(decode_pred)
        decode_label_list.append(label)

    result_dict = {}

    for metric_name, result in zip(['Acc', 'Precision', 'Recall', 'F1'],
                                   get_ner_fmeasure(decode_label_list,
                                                    decode_pred_list,
                                                    label_type='BIO')):
        print('%s Score: %f' % (metric_name,  result))
        result_dict[metric_name] = result
    return result_dict


def acc_evaluate(problem, estimator, params):
    text, label_data, label_encoder = params.read_data_fn[problem](
        params, PREDICT)
    t_l_tuple_list = list(zip(text, label_data))
    t_l_tuple_list = sorted(t_l_tuple_list, key=lambda t: len(t[0]))
    text, label_data = zip(*t_l_tuple_list)

    def pred_input_fn(): return predict_input_fn(text, params, mode=PREDICT)

    pred_list = estimator.predict(pred_input_fn)

    decode_pred_list = []
    decode_label_list = []

    scope_name = params.share_top[problem]

    for p, label, t in zip(pred_list, label_data, text):
        if not t:
            continue

        pred_prob = p[scope_name]

        if params.problem_type[problem] in ['seq_tag']:
            true_seq_length = len(t) - 1
            pred_prob = pred_prob[1:true_seq_length]

            # crf returns tags
            predict = pred_prob
            label = label[:len(predict)]
            assert len(pred_prob) == len(label), print(
                len(pred_prob), len(label))
            if not params.crf:
                predict = np.argmax(predict, axis=-1)
            decode_pred = label_encoder.inverse_transform(predict)
            decode_label = label
        elif params.problem_type[problem] in ['cls']:
            predict = np.argmax(pred_prob)

            decode_pred = label_encoder.inverse_transform([predict])
            decode_label = [label]
        elif params.problem_type[problem] in ['multi_cls']:
            predict = np.round(pred_prob)
            decode_pred = predict
            decode_label = label
        else:
            raise ValueError(
                'Acc evaluation dose not support problem type %s' % params.problem_type[problem])

        decode_pred_list.append(decode_pred)
        decode_label_list.append(decode_label)

    correct_char_count = 0
    correct_seq_count = 0
    total_char = 0

    for pred, label in zip(decode_pred_list, decode_label_list):
        total_char += len(label)
        if np.array(pred == label).all():
            correct_seq_count += 1
            correct_char_count += len(label)
        else:
            for pred_char, label_char in zip(pred, label):
                if pred_char == label_char:
                    correct_char_count += 1

    result_dict = {
        '%s_Accuracy' % problem: correct_char_count / total_char,
        '%s_Accuracy Per Sequence' % problem: correct_seq_count / len(decode_label_list)}

    return result_dict


def cws_evaluate(problem, estimator, params):
    text, label_data, label_encoder = params.read_data_fn[problem](
        params, PREDICT)
    t_l_tuple_list = list(zip(text, label_data))
    t_l_tuple_list = sorted(t_l_tuple_list, key=lambda t: len(t[0]))
    text, label_data = zip(*t_l_tuple_list)

    def pred_input_fn(): return predict_input_fn(text, params, mode=PREDICT)

    pred_list = estimator.predict(pred_input_fn)

    decode_pred_list = []
    decode_label_list = []

    scope_name = params.share_top[problem]

    for p, label, t in zip(pred_list, label_data, text):
        if not t:
            continue
        true_seq_length = len(t) - 1

        pred_prob = p[scope_name]

        pred_prob = pred_prob[1:true_seq_length]

        # crf returns tags
        predict = pred_prob
        label = label[:len(predict)]
        assert len(pred_prob) == len(label), print(len(pred_prob), len(label))

        if not params.crf:
            predict = np.argmax(predict, axis=-1)
        decode_pred = label_encoder.inverse_transform(predict)

        decode_pred_list.append(decode_pred)
        decode_label_list.append(label)

    result_dict = {}

    for metric_name, result in zip(['Acc', 'Precision', 'Recall', 'F1'],
                                   get_cws_fmeasure(decode_label_list,
                                                    decode_pred_list)):
        print('%s Score: %f' % (metric_name,  result))
        result_dict[metric_name] = result
    return result_dict


def get_cws_fmeasure(goldTagList, resTagList):
    scoreList = []
    assert len(resTagList) == len(goldTagList)

    # calculate tag wise acc
    total_length = 0
    for g in goldTagList:
        total_length += len(g)
    acc = np.sum(
        [
            np.sum(g == resTagList[i])
            for i, g in enumerate(goldTagList)]) / total_length

    getNewTagList(goldTagList)
    getNewTagList(resTagList)
    goldChunkList = getChunks(goldTagList)
    resChunkList = getChunks(resTagList)
    gold_chunk = 0
    res_chunk = 0
    correct_chunk = 0
    for i in range(len(goldChunkList)):
        res = resChunkList[i]
        gold = goldChunkList[i]
        resChunkAry = res.split(',')
        tmp = []
        for t in resChunkAry:
            if len(t) > 0:
                tmp.append(t)
        resChunkAry = tmp
        goldChunkAry = gold.split(',')
        tmp = []
        for t in goldChunkAry:
            if len(t) > 0:
                tmp.append(t)
        goldChunkAry = tmp
        gold_chunk += len(goldChunkAry)
        res_chunk += len(resChunkAry)
        goldChunkSet = set()
        for im in goldChunkAry:
            goldChunkSet.add(im)
        for im in resChunkAry:
            if im in goldChunkSet:
                correct_chunk += 1
    pre = correct_chunk / res_chunk
    rec = correct_chunk / gold_chunk
    f1 = 0 if correct_chunk == 0 else 2 * pre * rec / (pre + rec)
    scoreList.append(acc)
    scoreList.append(pre)
    scoreList.append(rec)
    scoreList.append(f1)

    infoList = []
    infoList.append(gold_chunk)
    infoList.append(res_chunk)
    infoList.append(correct_chunk)
    return scoreList


def getNewTagList(tagList):
    tmpList = []
    for im in tagList:
        tagAry = im
        newTags = ",".join(tagAry)
        tmpList.append(newTags)
    tagList.clear()
    for im in tmpList:
        tagList.append(im)


def getChunks(tagList):
    tmpList = []
    for im in tagList:
        tagAry = im.split(',')
        tmp = []
        for t in tagAry:
            if t != "":
                tmp.append(t)
        tagAry = tmp
        chunks = ""
        for i in range(len(tagAry)):
            if tagAry[i].upper().startswith("B"):
                pos = i
                length = 1
                ty = tagAry[i]
                for j in range(i + 1, len(tagAry)):
                    if tagAry[j].upper() in ["M", "E"]:
                        length += 1
                    else:
                        break
                chunk = ty + "*" + str(length) + "*" + str(pos)
                chunks = chunks + chunk + ","
        tmpList.append(chunks)
    return tmpList


def seq2seq_evaluate(problem, estimator, params):
    text, label_data, label_encoder = params.read_data_fn[problem](
        params, PREDICT)
    t_l_tuple_list = list(zip(text, label_data))
    t_l_tuple_list = sorted(t_l_tuple_list, key=lambda t: len(t[0]))
    text, label_data = zip(*t_l_tuple_list)

    def pred_input_fn(): return predict_input_fn(text, params, mode=PREDICT)

    pred_list = estimator.predict(pred_input_fn)

    decode_pred_list = []
    decode_label_list = []

    scope_name = params.share_top[problem]

    for p, label, t in zip(pred_list, label_data, text):
        if not t:
            continue

        pred_prob = p[scope_name]

        # crf returns tags
        predict = pred_prob

        decode_pred = [t for t in label_encoder.inverse_transform(
            predict) if t != '[PAD]']
        decode_label = [t for t in label if t != '[PAD]']

        decode_pred_list.append(decode_pred)
        decode_label_list.append([decode_label])

    result_dict = {}
    bleu1 = bleu_score.corpus_bleu(
        decode_label_list, decode_pred_list, weights=(1, 0, 0, 0))
    bleu4 = bleu_score.corpus_bleu(decode_label_list, decode_pred_list)

    for metric_name, result in zip(['BLEU1', 'BLEU4'],
                                   [bleu1, bleu4]):
        print('%s Score: %f' % (metric_name,  result))
        result_dict[metric_name] = result
    return result_dict
