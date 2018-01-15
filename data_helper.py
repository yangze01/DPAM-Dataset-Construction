#coding=utf8
import json
import sys
import os
import numpy as np
from optOnMysql.DocumentsOnMysql import *

reload(sys)
sys.setdefaultencoding('utf8')
BasePath = sys.path[0]


def read_from_json(path):
    with open(path, 'rb') as jd:
        data = json.loads(jd.read())
    return data


def get_dev_train_data(x_path, y_path):
    x_data = read_from_json(x_path)['x']
    print("finished get x_data")
    y_data = np.loadtxt(y_path)
    print("finished get y_data")
    # print(x_data)
    # print(y_data)
    return np.array(x_data), y_data

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



if __name__ == "__main__":
    criminal_list = ['交通肇事罪',  # 危险驾驶罪（危险 驾驶罪）
                     '过失致人死亡罪', # 故意杀人罪（故意 杀人 杀人罪） 故意伤害罪（故意 伤害 伤害罪）
                      '故意杀人罪',
                      '故意伤害罪',
                      '过失致人重伤罪',
                      '抢劫罪',
                      '诈骗罪', #（诈骗 诈骗罪 诈骗案）
                      '拐卖妇女儿童罪'
                      ]
    print(1)

    opt = DocumentsOnMysql()
    it = opt.exeQuery("show columns from document")
    columns_name_dict = dict()
    for id, iter in enumerate(it):
        columns_name_dict[id] = iter[0]

    criminal_id = 0
    for criminal in criminal_list:
        print(criminal)
        it = opt.findbycriminal(criminal)
        criminal_dir = BasePath + "/data/" + criminal.encode('utf8')
        if not os.path.exists(criminal_dir):
            os.makedirs(criminal_dir)
            print("make criminal {} dir".format(criminal_dir))
        else:
            print("criminal {} dir exists".format(criminal_dir))

        # print(criminal_dir)
        for index,data in enumerate(it):
            print(index)
            file_name = data[3] + "_" + data[1] + ".txt"
            content = '\n'.join(data[25].split('|'))
            file_path = criminal_dir + "/" + file_name
            try:
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        f.write(content)
            except:
                continue

        # for i,j in enumerate(it[0]):
        #     print("{}, {}\t\t\t\t\t{}".format(i, columns_name_dict[i], j))

