#coding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
BasePath = sys.path[0]
from data_helper import *

if __name__ == "__main__":
    # dict_path = BasePath + "/Segment/law_dict.txt"
    # with open(dict_path, 'rw') as f:
    #     data = f.read()
    #     print(data.split('\n').encode('utf8'))
    topn = 30
    train_dev_x_file_path = BasePath + "/data/all_train_dev_data_x.txt"
    train_dev_y_file_path = BasePath + "/data/all_train_dev_data_y_topn" + str(topn) + ".txt"
    x_text, y = get_dev_train_data(train_dev_x_file_path, train_dev_y_file_path)
    print(len(x_text)*0.1)


