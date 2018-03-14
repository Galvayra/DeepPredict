import math


class MyOneHotEncoder:
    def __init__(self):
        self.vector_dict = dict()

    # J : 연령, K : 성별, O : 주증상, AN : 의식, AO : 수축혈압, AP : 이완혈압, AQ : 맥박수, AR : 호흡수, AS : 체온
    def encoding(self, myData):
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

        def __set_scalar_dict__(value_list, except_start=0, except_end=0):
            scalar_dict = dict()
            scalar_list = list()

            for i in sorted(list(set(value_list))):
                if not math.isnan(i):
                    scalar_list.append(i)

            scalar_list = scalar_list[except_start:]

            for i in range(except_end):
                scalar_list.pop()

            scalar_dict["min"] = scalar_list[0]
            scalar_dict["max"] = scalar_list[-1]
            scalar_dict["div"] = float(scalar_dict["max"] - scalar_dict["min"])

            # print(scalar_list)
            # print(scalar_dict)

            return scalar_dict

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

        # show result of columns inspecting
        # k_dict = dict()
        #
        # for k, v in myData.data_dict.items():
        #
        #     type_dict = __inspect_column__(v)
        #     k_dict[k] = type_dict
        #
        # for k in sorted(k_dict.keys()):
        #     print(k, k_dict[k])

        scalar_columns = ['D', 'E', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Q', 'S', 'T', 'U', 'V', 'W', 'X',
                          'Y', 'Z', 'AA', 'AB', 'AC']
        scalar_columns_start = ['J', 'H']
        scalar_columns_end = ['K']
        scalar_columns_start_end = ['K']

        class_columns = ['P', 'R', 'AD']

        for k in sorted(myData.data_dict.keys()):

            if k in scalar_columns:
                self.vector_dict[k] = __set_scalar_dict__(myData.data_dict[k], except_start=0, except_end=0)
            elif k in scalar_columns_start:
                self.vector_dict[k] = __set_scalar_dict__(myData.data_dict[k], except_start=1, except_end=0)
            elif k in scalar_columns_end:
                self.vector_dict[k] = __set_scalar_dict__(myData.data_dict[k], except_start=0, except_end=1)
            elif k in scalar_columns_start_end:
                self.vector_dict[k] = __set_scalar_dict__(myData.data_dict[k], except_start=0, except_end=0)
            elif k in class_columns:
                self.vector_dict[k] = __set_class_dict__(myData.data_dict[k])

    def fit(self, data_dict, data_count):
        def _init_x_data():
            _x_data = list()

            # set X(number of rows) using rows_data
            # array dimension = X * Y(number of data)
            for _i in range(data_count):
                _x_data.append(list())

            return _x_data

        def _make_vector_from_class():
            for c in class_list:
                if c == value:
                    x_data[i].append(1.0)
                else:
                    x_data[i].append(0.0)

        x_data = _init_x_data()

        for k, v in data_dict.items():
            encode_dict = self.vector_dict[k]

            # key : 성별
            if k == "K":
                class_list = encode_dict.keys()
                for i, value in enumerate(v):
                    _make_vector_from_class()
            # key : 주증상
            elif k == "O":
                # pass
                class_list = encode_dict.keys()
                for i, value in enumerate(v):
                    _make_vector_from_class()
            # key : 의식
            elif k == "AN":
                class_list = encode_dict.keys()
                for i, value in enumerate(v):
                    _make_vector_from_class()
            # scalar vector
            else:
                minimum = encode_dict["min"]
                maximum = encode_dict["max"]
                division = encode_dict["div"]

                for i, value in enumerate(v):
                    # exception
                    if value < minimum or value > maximum:
                        value = float(-1)
                    # normalization
                    else:
                        value = (value - minimum)/division

                    x_data[i].append(value)

        return x_data
