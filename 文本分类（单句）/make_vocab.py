'''
Description: 构造字/词表，生成vocab文件
Author: Fu Guanyu
Date: 2020-11-16 15:29:49
LastEditTime: 2020-12-14 23:07:52
LastEditors: Please set LastEditors
Remarks: Copyright (c) 2019, Fu Guanyu. All rights reserved
'''
import json
from collections import Counter
from itertools import chain

def make_vocab(texts, VOCAB_SIZE, file_path='vocab', word2id_file_path='word2id'):
    '''
    Create the `vocab` file
    Args:
        texts: [[str, ...], ...]. A list of tokens-(list) which has been split by 'split_text'
        VOCAB_SIZE: int. Limit the size of vocabulary
        file_path: str. The path of `vocab` file, for example: 'data/vocab'
    '''
    words = chain.from_iterable(texts)
    vocab_counter = Counter(list(words))
    _, count = vocab_counter.most_common()[0]
    vocab_counter.update(['<PAD>']*(count*2+1))
    vocab_counter.update(['<UNK>']*(count+1))
    print("Number of init vocabulary: ", len(vocab_counter))
    word2id = {}
    with open(file_path, 'w', encoding='utf-8') as f, \
        open(word2id_file_path, 'w', encoding='utf-8') as wf:
        for word, count in vocab_counter.most_common(VOCAB_SIZE):
            if word != '':
                word2id[word] = len(word2id)
                f.write(word + ' ' + str(count) + '\n')
            else:
                repr('some thing wrong for ' + word)
        json.dump(word2id, wf, ensure_ascii=False, indent=4)
    return word2id

def filter_text(text):
    '''
    过滤掉一些在中文中无意义的特殊字符，譬如 \xa0 \u2002等有空白含义的字符用split可以去掉，
        对于其他的特殊字符仍需要继续完善
    Args:
        text: str. Initial text
    Returns:
        text: str. Filtered text
    '''
    return ''.join(text.split())

if __name__ == '__main__':
    make_vocab([['st', '信通', '遭大', '股东', '转嫁', '32', '亿', '债务', '大', '股东']], 10, 'vocab', 'word2id.json')
    

