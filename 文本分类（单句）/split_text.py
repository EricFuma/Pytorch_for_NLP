'''
Description: 
Author: Fu Guanyu
Date: 2020-11-15 15:28:44
LastEditTime: 2020-12-14 23:04:59
LastEditors: Please set LastEditors
Remarks: Copyright (c) 2019, Fu Guanyu. All rights reserved
'''
from jieba import lcut

'''
待改进： 譬如 123， 212.5 这样的实体是数字，应该用一类一个 token 表示这是 纯数字实体，譬如 <NUM>
'''


def str_to_char(text, is_lower=True):
    '''Split a string into a list of characters
    Args:
        text: str. string to be split
        is_lower: bool. Whether to ignore the case of English letters
    
    Returns:
        charlist: list. Store the split character list
    '''
    if is_lower:
        text = text.lower()
    
    num_set = set(['0','1','2','3','4','5','6','7','8','9','.','%'])
    char_set = set(['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m'])
    # Initializing charlist
    charlist = []

    length = len(text)
    number = ''
    char = ''
    for idx in range(length):
        if text[idx] in num_set:
            number += text[idx]
            if idx + 1 >= length or text[idx+1] not in num_set:
                charlist += [number]
                number = ''
        elif text[idx] in char_set:
            char += text[idx]
            if idx + 1 >= length or text[idx+1] not in char_set:
                charlist += [char]
                char = ''
        else:
            charlist += [text[idx]]
    return charlist
    

def str_to_word(text, is_lower=True):
    '''
    暂时使用 `jiabe分词` 作为中文词的工具，使用的是全分词模式
    Args:
        text: str. The string to be split
        is_lower: bool. Whether to ignore the case of English letter
    
    Returns:
        wordlist: list. Store the split word list
    '''
    if is_lower:
        text = text.lower()
    return lcut(text, cut_all=True)  # 使用 `jieba` 全模式进行分词

if __name__ == '__main__':
    print(str_to_char('st信通遭大股东转嫁32亿债务'))
    print(str_to_word('st信通遭大股东转嫁32亿债务'))
