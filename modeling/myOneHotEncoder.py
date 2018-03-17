# -*- coding: utf-8 -*-
import math
from .variables import *
from .options import *


# initial information & Past history 만을 이용하여 학습
class MyOneHotEncoder:
    def __init__(self):
        self.vector_dict = dict()

    def encoding(self, data_dict):
        def __inspect_column__(value_list):
            type_dict = dict()
            for i, value in enumerate(value_list):
                key = 0
                if type(value) is float:
                    if math.isnan(value):
                        continue
                    key = "float"
                elif type(value) is str:
                    key = "str"
                elif type(value) is int:
                    key = "int"

                if key not in type_dict:
                    type_dict[key] = 1
                else:
                    type_dict[key] += 1

            return type_dict

        # scalar dictionary 생성을 위해 앞 뒤 예외처리를 해야하는지 각 column 마다 확인해주어야 한다
        def __set_scalar_dict__(value_list, except_start=0, except_end=0):
            scalar_dict = dict()
            scalar_list = list()

            for i in sorted(list(set(value_list))):
                # 공백은 사전에 넣지 않음
                if not math.isnan(i):
                    scalar_list.append(i)

            scalar_list = scalar_list[except_start:]

            for i in range(except_end):
                scalar_list.pop()

            # 공백을 체크하는 영역
            # for i, value in enumerate(value_list):
            #     if math.isnan(value):
            #         print(k, i, value)

            scalar_dict["min"] = scalar_list[0]
            scalar_dict["max"] = scalar_list[-1]
            scalar_dict["div"] = float(scalar_dict["max"] - scalar_dict["min"])

            # print(k)
            # print(scalar_list)
            # print(scalar_dict)

            return scalar_dict

        # 셀의 공백은 type is not str 으로 찾을 수 있으며, 공백(nan)을 하나의 차원으로 볼지에 대한 선택을 우선 해야한다
        def __set_class_dict__(vector_list):
            class_dict = dict()

            for i, v in enumerate(vector_list):
                if v not in class_dict:
                    class_dict[v] = 0

                class_dict[v] += 1

            return class_dict
        #
        # def _set_symptom_dict():
        #     symptom_dict = dict()
        #
        #     for line in vector_list:
        #         line = line.strip()
        #         if line.endswith(";"):
        #             line = line[:-1]
        #
        #         for symptom in line.split(";"):
        #             symptom = symptom.strip()
        #
        #             if symptom not in symptom_dict:
        #                 symptom_dict[symptom] = 0
        #
        #             symptom_dict[symptom] += 1
        #
        #     return symptom_dict

        # # show result of columns inspecting
        # k_dict = dict()
        #
        # for k, v in data_dict.items():
        #
        #     type_dict = __inspect_column__(v)
        #     k_dict[k] = type_dict
        #
        # for k in sorted(k_dict.keys()):
        #     print(k, k_dict[k])

        for k in sorted(data_dict.keys()):
            if k in scalar_columns["0"]:
                self.vector_dict[k] = __set_scalar_dict__(data_dict[k], except_start=0, except_end=0)
            elif k in scalar_columns["start_1"]:
                self.vector_dict[k] = __set_scalar_dict__(data_dict[k], except_start=1, except_end=0)
            # elif k in scalar_columns["end_1"]:
            #     self.vector_dict[k] = __set_scalar_dict__(data_dict[k], except_start=0, except_end=1)
            elif k in scalar_columns["start_1_end_1"]:
                self.vector_dict[k] = __set_scalar_dict__(data_dict[k], except_start=1, except_end=1)
            elif k in class_columns:
                self.vector_dict[k] = __set_class_dict__(data_dict[k])

            # if k in class_columns:
            #     self.vector_dict[k] = __set_class_dict__(data_dict[k])

    def fit(self, data_dict, data_count):
        def ___init_x_data__():
            _x_data = list()

            # set X(number of rows) using rows_data
            # array dimension = X * Y(number of data)
            for _i in range(data_count):
                _x_data.append(list())

            return _x_data

        def __make_vector_use_scalar__():
            for _i, _value in enumerate(v):
                if math.isnan(_value):
                    _value = float(0)
                elif _value < minimum:
                    _value = float(MIN_SCALING)
                elif _value > maximum:
                    _value = float(1)
                # normalization
                else:
                    _value = (_value - minimum + MIN_SCALING)/(division + MIN_SCALING)

                x_data[_i].append(_value)

        def __make_vector_use_class__():
            for c in class_list:
                if c == value:
                    x_data[i].append(float(1))
                else:
                    x_data[i].append(float(0))

        def __get_all_columns__(columns_dict):
            _columns = list()
            for columns in columns_dict.values():
                _columns += columns

            return _columns

        x_data = ___init_x_data__()

        for k, v in data_dict.items():
            if k in __get_all_columns__(scalar_columns):
                encode_dict = self.vector_dict[k]
                minimum = encode_dict["min"]
                maximum = encode_dict["max"]
                division = encode_dict["div"]
                __make_vector_use_scalar__()
            elif k in class_columns:
                encode_dict = self.vector_dict[k]
                class_list = encode_dict.keys()
                for i, value in enumerate(v):
                    __make_vector_use_class__()

            # if k in class_columns:
            #     encode_dict = self.vector_dict[k]
            #     class_list = encode_dict.keys()
            #     for i, value in enumerate(v):
            #         __make_vector_use_class__()

        return x_data
