# -*- coding: utf-8 -*-
'''
@File    : check_lexicon.py
@Time    : 2023/10/14 10:28:03
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
MJSyntax:
- max length of label: N=23, 
- size of character set: C=37
'''

if __name__ == '__main__':
    pass

def get_info_from_lexicon(path_lexicon):
    """
    Return:
    - `N`: the maximum length of labels.
    - `C`: the size of character set.
    - `
    """
    max_len = 0
    char_set = set()
    with open(path_lexicon, 'r') as file:
        for line in file:
            word = line.strip()
            char_set.update(set(word.lower()))
            char_set.update(set(word.capitalize()))
            max_len = max(max_len, len(word))
    char_set = [0] + sorted(ord(c) for c in list(char_set))
    ch2idx = {char_set[i]: i for i in range(len(char_set))}
    idx2ch = dict(enumerate(char_set))
    return max_len, len(char_set), ch2idx, idx2ch


if __name__ == '__main__':
    path_lexicon = r'/home/wty/Coding/datasets/mjsynth/lexicon.txt'
    N, C, ch2idx, idx2ch = get_info_from_lexicon(path_lexicon)
    print(N, C, ch2idx, idx2ch)
