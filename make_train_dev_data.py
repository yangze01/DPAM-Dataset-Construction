#coding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import numpy as np

BasePath = sys.path[0]

class Vocab(object):
    def __init__(self, vocab_file):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n'%line)
                    continue
                w = pieces[0]
                count = pieces[1]
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                if int(w) == 0:
                    continue
                self._word_to_id[int(w)] = int(count)
                # self._id_to_word[self._count] = w
                self._count += 1
        print("Finished constructing vocabulary of %i total words. " % (self._count))

    def word2count(self, word):
        """Returns the id(integer) of a word (string). Returns [UNK] id if word is not in vocab"""
        return self._word_to_id[word]


if __name__ == "__main__":
    criminal_list = ['交通肇事罪',  # 危险驾驶罪（危险 驾驶罪）
                     '过失致人死亡罪', # 故意杀人罪（故意 杀人 杀人罪） 故意伤害罪（故意 伤害 伤害罪）
                      '故意杀人罪',
                      '故意伤害罪',
                      '过失致人重伤罪',
                      '抢劫罪',
                      #'诈骗罪', #（诈骗 诈骗罪 诈骗案）
                      '拐卖妇女儿童罪']

    for criminal in criminal_list:
        data_path = BasePath + "/data/" + criminal + "_raw/" + criminal + "_json_data.txt"
        vocab_path = BasePath + "/data/" + criminal + "_raw/" + criminal + "_vocab.txt"
        save_path = BasePath + "/data/raw_split_" + criminal + "_json_data.txt"

        # 将json_data保存为原始数据，即保存为案例对应的所有法条
        # save_data2rawdata(data_path, save_path)
        print(vocab_path)
        vocab = Vocab(vocab_path)
        # print("the vocab is : ")
        # print(vocab._word_to_id)

        sorted_vocab = sorted(vocab._word_to_id.items(), key = lambda e: e[1], reverse=True)
        print("the sorted vocab is : ")
        print(sorted_vocab)
        vocab_len = len(sorted_vocab)
        print("the len of sorted vocab is : {}".format(vocab_len))
        print("the len of max vocab is : {}".format(vocab_len/10*10))



