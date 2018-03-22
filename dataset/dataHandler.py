# -*- coding: utf-8 -*-
import pandas as pd
import math
from .variables import *


# ### refer to reference file ###
class DataHandler:
    def __init__(self):
        file_name = DATA_PATH + DATA_FILE
        self.rows_data = pd.read_csv(file_name)
        print("Read csv file -", file_name, "\n\n")
        self.file_name = DATA_FILE
        self.head_dict = {self.__get_head_dict_key__(i): v for i, v in enumerate(self.rows_data)}
        self.erase_index_list = self.__set_erase_index_list__()
        self.data_dict = self.__set_data_dict__()
        self.y_data = self.__set_labels__()
        self.__free__()

    def __get_head_dict_key__(self, index):

        alpha_dict = {
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
            10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
            20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
        }
        alpha_len = len(alpha_dict)
        key = str()

        if index < alpha_len:
            return alpha_dict[index]

        key_second = int(index / alpha_len) - 1
        key_first = index % alpha_len

        key += self.__get_head_dict_key__(key_second)
        key += alpha_dict[key_first]

        return key

    def __set_header_list__(self, start, end):
        return [self.__get_head_dict_key__(where) for where in range(start, end + 1)]

    def __set_data_dict__(self):

        data_dict = dict()

        header_list = ["A"]
        # ["D", ... ,"CP"], D=3, CP=93
        header_list += self.__set_header_list__(start=3, end=93)

        for header in header_list:
            header_key = self.head_dict[header]
            data_dict[header] = list()

            for i, data in enumerate(self.rows_data[header_key]):
                if i not in self.erase_index_list:
                    if type(data) is int:
                        data = float(data)
                    if type(data) is str:
                        data = data.strip()

                    data_dict[header].append(data)

        print("num of", len(data_dict[header]), "data!\n")
        return data_dict

    def __set_erase_index_list__(self):

        # header keys 조건이 모두 만족 할 때
        def __condition__(header_list, condition):
            header_keys = [self.head_dict[i] for i in header_list]

            _erase_index_dict = {i: 0 for i in range(len(self.rows_data[header_keys[0]]))}

            for header_key in header_keys:
                for index, value in enumerate(self.rows_data[header_key]):
                    value = str(value)

                    if condition == 0:
                        if value == str(condition) or value == str(0.0) or value == "nan":
                            _erase_index_dict[index] += 1
                    else:
                        if value == str(condition):
                            _erase_index_dict[index] += 1

            return _erase_index_dict, len(header_list)

        def __append__(_erase_index_dict, _num_match, _individual=False):
            for index, v in _erase_index_dict.items():
                if _individual and v >= _num_match:
                    if index not in erase_index_list:
                        erase_index_list.append(index)
                elif not _individual and v == _num_match:
                    if index not in erase_index_list:
                        erase_index_list.append(index)

        def __append_no_data__(header_key="F"):
            for index, v in self.rows_data[self.head_dict[header_key]].items():
                if type(v) is float:
                    if math.isnan(v):
                        erase_index_list.append(index)
                else:
                    if v == "N/V":
                        erase_index_list.append(index)

        def __cut_random_data__(_erase_index_list):
            r_num = int(CUT_RATIO.split('/')[1])
            cut_count = 0
            header_key = self.head_dict["DC"]
            for i, data in enumerate(self.rows_data[header_key]):
                if i not in _erase_index_list:
                    if data != "사망":
                        cut_count += 1
                        if cut_count % r_num == 0:
                            _erase_index_list.append(i)

        erase_index_list = list()

        # G : 수축혈압, H : 이완혈압, I : 맥박수, J : 호흡수 == 0 제외
        erase_index_dict, num_match = __condition__(header_list=["G", "H", "I", "J"], condition=0)
        __append__(erase_index_dict, num_match)

        erase_index_dict, num_match = __condition__(header_list=["G", "H", "I", "J"], condition=-1)
        __append__(erase_index_dict, num_match)

        # # 주증상 데이터가 없는 경우
        # __append_no_data__()
        #
        # # 혈액관련 데이터가 없는 경우
        # erase_index_dict, num_match = __condition__(header_list=["AE", "AF", "AG", "AH", "AI", "AM", "AN",
        #                                                          "AO", "AQ", "AR", "AS", "AT", "AU", "AV", "AW", "AX",
        #                                                          "AY", "BC", "BD", "BE", "BF", "BG", "BH", "BK", "BL"
        #                                                          ], condition=0)
        __append__(erase_index_dict, 1, _individual=True)

        # __cut_random_data__(erase_index_list)

        print("num of", len(erase_index_list), "data is excepted!\n")

        return sorted(erase_index_list, reverse=True)

    # DC : 퇴원형태
    def __set_labels__(self):
        y_labels = list()

        header_key = self.head_dict["DC"]

        for i, value in enumerate(self.rows_data[header_key]):
            if i not in self.erase_index_list:
                if value == "사망":
                    y_labels.append([1])
                else:
                    y_labels.append([0])

        return y_labels

    def __free__(self):
        del self.rows_data
        del self.head_dict
        del self.erase_index_list

    def counting_mortality(self, data):
        count = 0
        for i in data:
            if i == [1]:
                count += 1

        return count
