# -*- coding: utf-8 -*-
import math
import gensim.models.keyedvectors as word2vec
from .variables import *
from .options import *


# initial information & Past history 만을 이용하여 학습
class MyOneHotEncoder:
    def __init__(self, w2v=True):
        self.vector_dict = dict()
        if w2v:
            self.model = word2vec.KeyedVectors.load_word2vec_format(DUMP_PATH + LOAD_WORD2VEC, binary=True)
            print("Read w2v file -", DUMP_PATH + LOAD_WORD2VEC)

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
                    for _i, v in enumerate(data_dict[_k]):
                        try:
                            v = float(v)
                        except ValueError:
                            print(_k, v, "(", _i, ")")

        # scalar dictionary 생성을 위해 앞 뒤 예외처리를 해야하는지 각 column 마다 확인해주어야 한다
        def __set_scalar_dict__(value_list, except_start=0, except_end=0):
            scalar_dict = dict()
            scalar_list = list()

            for _i in sorted(list(set(value_list))):
                # 공백은 사전에 넣지 않음
                if not math.isnan(_i):
                    scalar_list.append(_i)

            scalar_list = scalar_list[except_start:]

            for _i in range(except_end):
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

            class_dict = dict()

            for _i, v in enumerate(vector_list):
                v = str(v).strip()

                # key exception is nan
                if v != "nan":
                    if self.__is_zero__(v):
                        v = str(0)

                    if v not in class_dict:
                        class_dict[v] = 1
                    else:
                        class_dict[v] += 1

            return class_dict

        def __set_symptom_dict__(vector_list):
            symptom_dict = dict()

            _input = input("Input(EXIT) -")

            while _input != "EXIT":

                try:
                    print(self.model.wv.most_similar(positive=[_input]))
                except KeyError:
                    print("not in vocab")

                _input = input("Input(EXIT) -")
            # for line in vector_list:
            #     line = line.strip()
            #     if line.endswith(";"):
            #         line = line[:-1]
            #
            #     for symptom in line.split(";"):
            #         symptom = symptom.strip()
            #
            #         if symptom not in symptom_dict:
            #             symptom_dict[symptom] = 0
            #
            #         symptom_dict[symptom] += 1

            return symptom_dict

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

            # elif columns_key == "word":
            #     if k in columns[columns_key]:
            #         self.vector_dict[k] = __set_symptom_dict__(data_dict[k])

        for k in sorted(data_dict.keys()):
            for columns in columns_dict.values():
                for columns_key in columns:
                    __set_vector_dict__()

    def __set_scalar_value_list__(self, key, value_list):

        new_value_list = list()

        for _i, value in enumerate(value_list):
            try:
                new_value_list.append(float(value))
            except ValueError:
                # 측정불가와 셀의 값이 0인것
                if value == "측정불가" or self.__is_zero__(value):
                    new_value_list.append(float(0))
                else:
                    ch = value[0]
                    if ch == ">" or ch == "<":
                        new_value_list.append(float(value[1:]))
                    else:
                        if value.find("이상") > 0:
                            v = value.split("이상")[0].strip()
                            new_value_list.append(float(v))
                        elif value.find("이하") > 0:
                            v = value.split("이하")[0].strip()
                            new_value_list.append(float(v))
                        else:
                            v = value.split("(")[0].strip()
                            new_value_list.append(float(v))

        return new_value_list

    def __is_zero__(self, string):
        is_zero = True
        for ch in string:
            if ch != '.':
                is_zero = False
        return is_zero

    def fit(self, data_dict, data_count):
        def __init_x_vector_dict__():
            _x_vector_dict = OrderedDict()
            _x_vector_dict["merge"] = list()
            for _columns_key in columns_dict:
                _x_vector_dict[_columns_key] = list()

            # set X(number of rows) using rows_data
            # array dimension = X * Y(number of data)
            for _key in _x_vector_dict:
                for _i in range(data_count):
                    _x_vector_dict[_key].append(list())

            return _x_vector_dict

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

                x_vector_dict["merge"][_i].append(_value)
                x_vector_dict[columns_key][_i].append(_value)

        def __make_vector_use_class__():

            _value = str(value).strip()
            if self.__is_zero__(_value):
                _value = str(0)

            for c in class_list:
                if c == _value:
                    x_vector_dict["merge"][i].append(float(1))
                    x_vector_dict[columns_key][i].append(float(1))
                else:
                    x_vector_dict["merge"][i].append(float(0))
                    x_vector_dict[columns_key][i].append(float(0))

        def __get_all_columns__(_columns_dict):
            all_columns = list()
            for _columns in _columns_dict.values():
                all_columns += _columns

            return all_columns

        x_vector_dict = __init_x_vector_dict__()

        for k, v in data_dict.items():
            for columns_key, columns in columns_dict.items():
                for columns_type_key in columns:
                    if columns_type_key == "scalar":
                        if k in __get_all_columns__(columns[columns_type_key]):
                            encode_dict = self.vector_dict[k]
                            minimum = encode_dict["min"]
                            maximum = encode_dict["max"]
                            division = encode_dict["div"]
                            __make_vector_use_scalar__()
                    elif columns_type_key == "class":
                        if k in columns[columns_type_key]:
                            encode_dict = self.vector_dict[k]
                            class_list = encode_dict.keys()
                            for i, value in enumerate(v):
                                __make_vector_use_class__()

        return x_vector_dict
