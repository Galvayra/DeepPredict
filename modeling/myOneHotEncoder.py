# -*- coding: utf-8 -*-
import math
from .variables import *
from .options import *


# initial information & Past history 만을 이용하여 학습
class MyOneHotEncoder:
    def __init__(self):
        self.vector_dict = dict()

    def encoding(self, data_dict):
        def __inspect_columns__():
            # show result of columns inspecting
            k_dict = dict()

            for _k, v in data_dict.items():
                type_dict = dict()

                for i, value in enumerate(data_dict[_k]):
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

                k_dict[_k] = type_dict

            for _k in sorted(k_dict.keys()):
                print(_k, k_dict[_k])

        # columns 이 전부 float 형에 어긋나는 행과 데이터를 출력
        def __inspect_float_column__():
            for _k in sorted(data_dict.keys()):
                if _k in blood_count_columns['scalar']:
                    for i, v in enumerate(data_dict[_k]):
                        try:
                            v = float(v)
                        except ValueError:
                            print(_k, v, "(", i, ")")

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

            scalar_dict["min"] = scalar_list[0]
            scalar_dict["max"] = scalar_list[-1]
            scalar_dict["div"] = float(scalar_dict["max"] - scalar_dict["min"])

            # print("\n" + k)
            # print(scalar_list)
            # print(scalar_dict)

            return scalar_dict

        # 셀의 공백은 type is not str 으로 찾을 수 있으며, 공백(nan)을 하나의 차원으로 볼지에 대한 선택을 우선 해야한다
        def __set_class_dict__(vector_list):
            def __is_zero__(string):
                is_zero = True
                for ch in string:
                    if ch != '.':
                        is_zero = False
                return is_zero

            class_dict = dict()

            for i, v in enumerate(vector_list):
                v = str(v).strip()

                # key exception is nan
                if v != "nan":
                    if __is_zero__(v):
                        v = str(0)

                    if v not in class_dict:
                        class_dict[v] = 1
                    else:
                        class_dict[v] += 1

            return class_dict


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

        # __inspect_columns__()
        # __inspect_float_column__()

        def __set_vector_dict__():
            def __get_scalar_key__(_key):
                _key_list = _key.split('#')

                if len(_key_list) == 2:
                    _start = int(_key_list[0].split('_')[1])
                    _end = int(_key_list[1].split('_')[1])
                else:
                    _key_list = _key_list[0].split('_')

                    if len(_key_list) == 2:
                        if _key_list[0] == "start":
                            _start = int(_key_list[1])
                            _end = 0
                        else:
                            _start = 0
                            _end = int(_key_list[1])
                    else:
                        _start = 0
                        _end = 0

                return _start, _end

            if columns_key == "scalar":
                for scalar_key, scalar_columns in columns[columns_key].items():
                    start, end = __get_scalar_key__(scalar_key)
                    if k in columns[columns_key][scalar_key]:
                        v_list = self.__set_scalar_value_list__(k, data_dict[k])
                        self.vector_dict[k] = __set_scalar_dict__(v_list, except_start=start, except_end=end)

            elif columns_key == "class":
                if k in columns[columns_key]:
                    self.vector_dict[k] = __set_class_dict__(data_dict[k])

        for k in sorted(data_dict.keys()):
            for columns in columns_dict.values():
                for columns_key in columns:
                    __set_vector_dict__()

    def __set_scalar_value_list__(self, key, value_list):
        new_value_list = list()

        for i, value in enumerate(value_list):
            try:
                new_value_list.append(float(value))
            except ValueError:
                if value == "측정불가":
                    new_value_list.append(float(0))
                    # print(key, i, value)
                else:
                    print(key, i, value)
                    # ch = value[0]
                    # if ch == ">" or ch == "<":
                    #     print(key, i, value)
                    # pass

        return new_value_list

    def fit(self, data_dict, data_count):
        def __init_x_vector__():
            _x_vector = list()

            # set X(number of rows) using rows_data
            # array dimension = X * Y(number of data)
            for _i in range(data_count):
                _x_vector.append(list())

            return _x_vector

        def __make_vector_use_scalar__():
            value_list = self.__set_scalar_value_list__(k, v)
            for _i, _value in enumerate(value_list):
                # type is float
                if math.isnan(_value):
                    _value = float(0)
                elif _value < minimum:
                    _value = float(MIN_SCALING)
                elif _value > maximum:
                    _value = float(1)
                # normalization
                else:
                    _value = (_value - minimum + MIN_SCALING)/(division + MIN_SCALING)

                x_vector[_i].append(_value)

        def __make_vector_use_class__():
            def __is_zero__(string):
                is_zero = True
                for ch in string:
                    if ch != '.':
                        is_zero = False
                return is_zero

            _value = str(value).strip()
            if __is_zero__(_value):
                _value = str(0)
                
            for c in class_list:
                if c == _value:
                    x_vector[i].append(float(1))
                else:
                    x_vector[i].append(float(0))

        def __get_all_columns__(_columns_dict):
            all_columns = list()
            for _columns in _columns_dict.values():
                all_columns += _columns

            return all_columns

        x_vector = __init_x_vector__()

        for k, v in data_dict.items():
            for columns in columns_dict.values():
                for columns_key in columns:
                    if columns_key == "scalar":
                        if k in __get_all_columns__(columns[columns_key]):
                            encode_dict = self.vector_dict[k]
                            minimum = encode_dict["min"]
                            maximum = encode_dict["max"]
                            division = encode_dict["div"]
                            __make_vector_use_scalar__()
                    elif columns_key == "class":
                        if k in columns[columns_key]:
                            encode_dict = self.vector_dict[k]
                            class_list = encode_dict.keys()
                            for i, value in enumerate(v):
                                __make_vector_use_class__()

        return x_vector
