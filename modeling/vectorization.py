from .myOneHotEncoder import MyOneHotEncoder
from .variables import DUMP_FILE, DUMP_PATH
from .options import IS_CLOSED, NUM_FOLDS, RATIO
from collections import OrderedDict
import json


class MyVector:
    def __init__(self, my_data):
        def __init_vector_list__():
            vector_dict = OrderedDict()
            # vector_dict["x_train"] = list()
            # vector_dict["y_train"] = list()
            # vector_dict["x_test"] = list()
            # vector_dict["y_test"] = list()
            vector_dict["x_train"] = dict()
            vector_dict["y_train"] = list()
            vector_dict["x_test"] = dict()
            vector_dict["y_test"] = list()

            if IS_CLOSED:
                return [vector_dict]
            else:
                return [vector_dict for i in range(NUM_FOLDS)]

        self.my_data = my_data
        self.file_name = my_data.file_name.split('.')[0]
        self.vector_list = __init_vector_list__()
        self.__set_vector_list__()
        self.__free__()

    def __set_vector_list__(self):
        def __set_x_data_dict__(is_manual=False, is_test=False):
            x_dict = dict()

            if is_manual:
                if is_test:
                    for _k, _vector_list in x_data_dict.items():
                        x_dict[_k] = _vector_list[:subset_size]
                else:
                    for _k, _vector_list in x_data_dict.items():
                        x_dict[_k] = _vector_list[subset_size:]
            else:
                if is_test:
                    for _k, _vector_list in x_data_dict.items():
                        x_dict[_k] = _vector_list[i * subset_size:][:subset_size]
                else:
                    for _k, _vector_list in x_data_dict.items():
                        x_dict[_k] = _vector_list[:i * subset_size] + _vector_list[(i + 1) * subset_size:]

            return x_dict

        # copy DataHandler to local variables
        x_data_dict = self.my_data.data_dict
        y_data = self.my_data.y_data[:]

        # init encoder
        my_encoder = MyOneHotEncoder(w2v=False)
        my_encoder.encoding(x_data_dict)

        # fit encoder into data
        if IS_CLOSED:
            self.vector_list[0]["y_train"] = y_data
            self.vector_list[0]["y_test"] = y_data
            num_train = len(self.vector_list[0]["y_train"])
            num_test = len(self.vector_list[0]["y_test"])
            self.vector_list[0]["x_train"] = my_encoder.fit(x_data_dict, num_train)
            self.vector_list[0]["x_test"] = my_encoder.fit(x_data_dict, num_test)
        else:
            num_folds = len(self.vector_list)

            # k-fold validation
            if num_folds > 1:
                subset_size = int(len(y_data) / NUM_FOLDS) + 1

                for i in range(num_folds):
                    self.vector_list[i]["y_train"] = y_data[:i * subset_size] + y_data[(i + 1) * subset_size:]
                    self.vector_list[i]["y_test"] = y_data[i * subset_size:][:subset_size]
                    num_train = len(self.vector_list[i]["y_train"])
                    num_test = len(self.vector_list[i]["y_test"])
                    self.vector_list[i]["x_train"] = my_encoder.fit(__set_x_data_dict__(), num_train)
                    self.vector_list[i]["x_test"] = my_encoder.fit(__set_x_data_dict__(is_test=True), num_test)
            # one fold
            else:
                subset_size = int(len(y_data) / RATIO)

                self.vector_list[0]["y_train"] = y_data[subset_size:]
                self.vector_list[0]["y_test"] = y_data[:subset_size]
                num_train = len(self.vector_list[0]["y_train"])
                num_test = len(self.vector_list[0]["y_test"])
                self.vector_list[0]["x_train"] = my_encoder.fit(__set_x_data_dict__(is_manual=True), num_train)
                self.vector_list[0]["x_test"] = my_encoder.fit(__set_x_data_dict__(is_manual=True,
                                                                                            is_test=True), num_test)

    def __free__(self):
        del self.my_data

    def dump(self):

        if IS_CLOSED:
            file_name = DUMP_PATH + DUMP_FILE + '_' + self.file_name + "_closed"
        else:
            file_name = DUMP_PATH + DUMP_FILE + '_' + self.file_name + "_opened_" + str(NUM_FOLDS)

        with open(file_name, 'w') as outfile:
            json.dump(self.vector_list, outfile, indent=4)
            print("success make dump file! - file name is", file_name)

