#coding=utf8
import json
import sys
import os
from optOnMysql.DocumentsOnMysql import *
reload(sys)
sys.setdefaultencoding('utf8')
BasePath = sys.path[0]



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

