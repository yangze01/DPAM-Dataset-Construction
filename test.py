#coding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
BasePath = sys.path[0]


if __name__ == "__main__":
    dict_path = BasePath + "/Segment/law_dict.txt"
    with open(dict_path, 'rw') as f:
        data = f.read()
        print(data.split('\n').encode('utf8'))





