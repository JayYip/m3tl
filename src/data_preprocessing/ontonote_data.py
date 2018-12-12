import copy


def get_all_word_labels(row: str):
    tag_buffer = []
    word_list = []
    tag_list = []
    for word in row.split():

        # add left bracket and tag
        if '(' in word:
            tag_buffer.append(word.replace('(', ''))

        else:
            print(tag_list)
            # append all tags in buffer to tag list
            assert ')' in word
            word_list.append(word.replace(')', ''))
            tag_list.append(copy.copy(tag_buffer))
            # remove the last tag based on how many right bracket
            for _ in range(word.count(')')):
                tag_buffer.pop()
    pos_target = [sublist[-1] for sublist in tag_list]
    ner_target = [sublist[-2] if 'NER' in sublist[-2]
                  else 'O' for sublist in tag_list]
