#coding=utf8
# import tensorflow as tf
import numpy as np
import os
import sys
import re
from Segment.MySegment import *
import jieba
import json
reload(sys)
sys.setdefaultencoding('utf8')
import os
BasePath = sys.path[0] + "/data/"

chs_arabic_map = {u'零':0, u'一':1, u'二':2, u'三':3, u'四':4,
        u'五':5, u'六':6, u'七':7, u'八':8, u'九':9,
        u'十':10, u'百':100, u'千':10 ** 3, u'万':10 ** 4,
        u'〇':0, u'壹':1, u'贰':2, u'叁':3, u'肆':4,
        u'伍':5, u'陆':6, u'柒':7, u'捌':8, u'玖':9,
        u'拾':10, u'佰':100, u'仟':10 ** 3, u'萬':10 ** 4,
        u'亿':10 ** 8, u'億':10 ** 8, u'幺': 1,
        u'０':0, u'１':1, u'２':2, u'３':3, u'４':4,
        u'５':5, u'６':6, u'７':7, u'８':8, u'９':9}
def make_count(reason_list, save_path):
    word_dict = dict()
    with open(save_path,'wb') as wf:
        for reason in reason_list:
            if reason not in word_dict:
                 word_dict[reason] = 1
            else:
                word_dict[reason] += 1
        for key in word_dict:
            print(key, word_dict[key])
            wf.write(str(key) + ' ' + str(word_dict[key]) + '\n')


class dataHelper(object):

    def __init__(self):
        print(1)

    def get_content(self, sentence):
        pattern_divided = re.compile('(.*)(依据|依照)《中华人民共和国刑法》')
        search_result = re.search(pattern_divided, sentence)
        if search_result:
            return search_result.group()

    def get_result(self, sentence):
        pattern_divided = re.compile('(依据|依照)《中华人民共和国刑法》(.*?)(《|。)')
        search_result = re.search(pattern_divided, sentence)
        if search_result:
            return search_result.group()

    def get_case_reason(self, sentence):
        pattern_fa_tiao3 = re.compile("第([零 一 二 三 四 五 六 七 八 九 十 百][零 一 二 三 四 五 六 七 八 九 十 百].*?)条")
        search_result = re.findall(pattern_fa_tiao3, sentence)
        if search_result:
            return search_result
        else:
            return -1
def chinese2num (chinese_digits, encoding="utf-8"):
    if isinstance (chinese_digits, str):
        chinese_digits = chinese_digits.decode (encoding)

    result  = 0
    tmp     = 0
    hnd_mln = 0
    for count in range(len(chinese_digits)):
        curr_char  = chinese_digits[count]
        curr_digit = chs_arabic_map.get(curr_char, None)
        # meet 「亿」 or 「億」
        if curr_digit == 10 ** 8:
            result  = result + tmp
            result  = result * curr_digit
            # get result before 「亿」 and store it into hnd_mln
            # reset `result`
            hnd_mln = hnd_mln * 10 ** 8 + result
            result  = 0
            tmp     = 0
        # meet 「万」 or 「萬」
        elif curr_digit == 10 ** 4:
            result = result + tmp
            result = result * curr_digit
            tmp    = 0
        # meet 「十」, 「百」, 「千」 or their traditional version
        elif curr_digit >= 10:
            tmp    = 1 if tmp == 0 else tmp
            result = result + curr_digit * tmp
            tmp    = 0
        # meet single digit
        elif curr_digit is not None:
            tmp = tmp * 10 + curr_digit
        else:
            return result
    result = result + tmp
    result = result + hnd_mln
    return result

if __name__ == "__main__":

    criminal_list = ['交通肇事罪',  # 危险驾驶罪（危险 驾驶罪）
                     '过失致人死亡罪', # 故意杀人罪（故意 杀人 杀人罪） 故意伤害罪（故意 伤害 伤害罪）
                      '故意杀人罪',
                      '故意伤害罪',
                      '过失致人重伤罪',
                      '抢劫罪',
                      #'诈骗罪', #（诈骗 诈骗罪 诈骗案）
                      '拐卖妇女儿童罪']

    myseg = MySegment()
    dataH = dataHelper()

    case_dict = dict()
    save_list = list()
    reason_list = list()

    for criminal in criminal_list:
        print("~~~~~~~~~~~~~~~~~~{}~~~~~~~~~~~~~~~~~~".format(criminal))
        # case_dict = dict()
        # save_list = list()
        # reason_list = list()

        file_dir = BasePath + criminal + "/"
        dir_list = os.listdir(file_dir)

        i = 0
        for dir_name in dir_list:
            # print(dir_name)
            with open(file_dir + dir_name, 'r') as f:
                try:
                    data = f.read()
                except:
                    print("error")
            title_list = list(myseg.sen2word(dir_name))
            # print(' '.join(title_list))
            if u'一审' in title_list or u'二审' in title_list:
                # print(i)
                # i += 1

                re_data = ''.join(data.split('\n')[3:])
                # print(re_data)
                content = dataH.get_content(re_data)
                result = dataH.get_result(re_data)
                # print(content)
                # print(result)
                if result:
                    case_reason = dataH.get_case_reason(result)
                    if case_reason != -1:
                        tmp_dict = dict()
                        print("------------------" + str(i) + "---------------------")
                        i += 1
                        cut_content_ltp = myseg.sen2word((content.encode('utf8')))
                        num_reason = [chinese2num(chn) for chn in case_reason if chinese2num(chn) != 0]
                        print(num_reason)
                        tmp_dict['content'] = cut_content_ltp
                        tmp_dict['result'] = num_reason
                        save_list.append(tmp_dict)
                        reason_list += num_reason
                    else:
                        print("can't catch the result")
                        # print(re_data)

        result_dir = BasePath
        # result_dir = BasePath + criminal + "_raw"

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            print("make criminal {} raw dir".format(result_dir))
        else:
            print("criminal {} raw dir exists".format(result_dir))

        # make_count(reason_list, result_dir + "/" + criminal + "_vocab.txt")
        # json_data = json.dumps(save_list, ensure_ascii=False)
        # print("the len of the data is : " + str(len(save_list)))
        # with open(result_dir + "/" + criminal + "_json_data.txt", 'wb') as f:
        #     f.write(json_data)

    make_count(reason_list, result_dir + "all_vocab.txt")
    json_data = json.dumps(save_list, ensure_ascii=False)
    print("the len of the data is : " + str(len(save_list)))
    with open(result_dir + "all_json_data.txt", 'wb') as f:
        f.write(json_data)








