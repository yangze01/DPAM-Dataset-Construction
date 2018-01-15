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

def one_hot_Vocab(keys):
    one_dict = dict()
    _count = 0
    for key in keys:
        if key in one_dict:
            raise Exception("Duplicated word in vocabulary file: %s" %key)
        if key == 0:
            continue
        one_dict[key] = _count
        _count += 1
    return one_dict

def read_from_json(path_file):
    with open(path_file, 'rb') as jd:
        data = json.loads(jd.read())
    return data

def writejson2file(dict_data, save_path):
    json_data = json.dumps(dict_data, ensure_ascii=False)
    with open(save_path, "wb") as f:
        f.write(json_data)


def save_data2rawdata(data_path, save_path):
    data = read_from_json(data_path)
    save_dict = dict()
    return_x = list()
    return_y = list()
    for line in data:
        return_x.append(' '.join(line['content']))
        return_y.append(line['result'])

    save_dict['content'] = return_x
    save_dict['result'] = return_y

    assert len(save_dict['content']) == len(save_dict['result']), "the len does not match"
    with open(save_path, 'wb') as rsjd:
        json_data = json.dumps(save_dict)
        rsjd.write(json_data)
    return save_dict

if __name__ == "__main__":
    criminal_list = ['交通肇事罪',  # 危险驾驶罪（危险 驾驶罪）
                     '过失致人死亡罪', # 故意杀人罪（故意 杀人 杀人罪） 故意伤害罪（故意 伤害 伤害罪）
                      '故意杀人罪',
                      '故意伤害罪',
                      '过失致人重伤罪',
                      '抢劫罪',
                      #'诈骗罪', #（诈骗 诈骗罪 诈骗案）
                      '拐卖妇女儿童罪']

    # for criminal in criminal_list:
    #     data_path = BasePath + "/data/" + criminal + "_raw/" + criminal + "_json_data.txt"
    #     vocab_path = BasePath + "/data/" + criminal + "_raw/" + criminal + "_vocab.txt"
    #     save_path = BasePath + "/data/" + criminal + "_raw/" + "raw_split_" + criminal + "_json_data.txt"
    #
    #     # 将json_data保存为原始数据，即保存为案例对应的所有法条
    #     data = save_data2rawdata(data_path, save_path)
    #
    #     vocab = Vocab(vocab_path)
    #     # print("the vocab is : ")
    #     # print(vocab._word_to_id)
    #
    #     sorted_vocab = sorted(vocab._word_to_id.items(), key = lambda e: e[1], reverse=True)
    #     print("the sorted vocab is : ")
    #     print(sorted_vocab)
    #     vocab_len = len(sorted_vocab)
    #     print("the len of sorted vocab is : {}".format(vocab_len))
    #     max_len = vocab_len/10*10
    #     for topn in range(10,max_len+1,10):
    #         keys = [tuple[0] for tuple in sorted_vocab[:topn]]
    #         # print("the keys set len is : {}".format(len(keys)))
    #         # print(keys)
    #         one_hot_vocab = one_hot_Vocab(keys)
    #         # print("the one hot dict is: ")
    #         # print(one_hot_vocab)
    #         with open(BasePath + "/data/" + criminal +
    #           "_raw/one_hot_vocab_" + criminal + str(topn) + ".txt", 'wb') as f:
    #             f.write(json.dumps(one_hot_vocab))
    #
    #         result_y = [list(set(result) & set(keys)) for result in data['result']]
    #         return_x = [' '.join(sentence) for sentence in data['content']]
    #         return_x = data['content']
    #         return_y = list()
    #         for line in result_y:
    #             y_test = np.array([0] * len(keys))
    #             one_hot = [one_hot_vocab[word] for word in line]
    #             y_test[one_hot] = 1
    #             return_y.append(y_test)
    #
    #         # train_dev_x_file_path = BasePath + "/data/train_dev_data_x.txt"
    #         train_dev_y_file_path = BasePath + "/data/" + criminal + "_raw/"+ criminal +"_train_dev_data_y_topn" + str(topn) + ".txt"
    #         np.savetxt(train_dev_y_file_path, np.array(return_y, dtype = np.int32))
    ##################################################################
    # data_path = BasePath + "/data/" + "all_json_data.txt"
    # vocab_path = BasePath + "/data/" + "all_vocab.txt"
    # save_path = BasePath + "/data/" + "all_raw_split_json_data.txt"
    #
    # # 将json_data保存为原始数据，即保存为案例对应的所有法条
    # data = save_data2rawdata(data_path, save_path)
    #
    # vocab = Vocab(vocab_path)
    # print("the vocab is: ")
    # print(vocab._word_to_id)
    #
    # sorted_vocab = sorted(vocab._word_to_id.items(), key = lambda e : e[1], reverse = True)
    # print("the sorted vocab is : ")
    # print(sorted_vocab)
    # vocab_len = len(sorted_vocab)
    # # print("the len of sorted vocabv is : {}".format(vocab_len))
    # max_len = vocab_len/10*10
    # return_x = [' '.join(sentence) for sentence in data['content']]
    # for topn in range(10, 30+1, 10):
    #     keys = [tuple[0] for tuple in sorted_vocab[:topn]]
    #     print("the keys set len is : {}".format(len(keys)))
    #     print(keys)
    #     one_hot_vocab = one_hot_Vocab(keys)
    #     print("the one hot dict is: ")
    #     print(one_hot_vocab)
    #     with open(BasePath + "/data/one_hot_vocab_" + str(topn) + ".txt", 'wb') as f:
    #         f.write(json.dumps(one_hot_vocab))
    #
    #     result_y = [list(set(result) & set(keys)) for result in data['result']]
    #     return_y = list()
    #     for line in result_y:
    #         y_test = np.array([0] * len(keys))
    #         one_hot = [one_hot_vocab[word] for word in line]
    #         y_test[one_hot] = 1
    #         return_y.append(y_test)
    #     train_dev_y_file_path = BasePath + "/data/all_train_dev_data_y_topn" + str(topn) + ".txt"
    #     np.savetxt(train_dev_y_file_path, np.array(return_y, dtype = np.int32))
    #
    # train_dev_x_file_path = BasePath + "/data/all_train_dev_data_x.txt"
    # writejson2file({'x': return_x}, train_dev_x_file_path)
