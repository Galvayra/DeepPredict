from .myOneHotEncoder import MyOneHotEncoder
from .variables import DUMP_FILE, DUMP_PATH
from .options import IS_CLOSED, NUM_FOLDS, RATIO, USE_W2V
from collections import OrderedDict
import json


class MyVector:
    def __init__(self, my_data):
        self.my_data = my_data
        self.file_name = my_data.file_name.split('.')[0]
        self.vector_list = list()

    def encoding(self):
        def __init_vector_dict__():
            vector_dict = OrderedDict()
            vector_dict["x_train"] = x_train
            vector_dict["y_train"] = y_train
            vector_dict["x_test"] = x_test
            vector_dict["y_test"] = y_test

            return vector_dict

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
        my_encoder = MyOneHotEncoder(w2v=USE_W2V)
        my_encoder.encoding(x_data_dict)

        # fit encoder into data
        if IS_CLOSED:
            y_train = y_data
            y_test = y_data
            x_train = my_encoder.fit(__set_x_data_dict__(), len(y_train))
            x_test = my_encoder.fit(__set_x_data_dict__(is_test=True), len(y_test))
            self.vector_list.append(__init_vector_dict__())
        else:
            # k-fold validation
            if NUM_FOLDS > 1:
                subset_size = int(len(y_data) / NUM_FOLDS) + 1

                for i in range(NUM_FOLDS):
                    y_train = y_data[:i * subset_size] + y_data[(i + 1) * subset_size:]
                    y_test = y_data[i * subset_size:][:subset_size]
                    x_train = my_encoder.fit(__set_x_data_dict__(), len(y_train))
                    x_test = my_encoder.fit(__set_x_data_dict__(is_test=True), len(y_test))
                    self.vector_list.append(__init_vector_dict__())

            # one fold
            else:
                subset_size = int(len(y_data) / RATIO)

                y_train = y_data[subset_size:]
                y_test = y_data[:subset_size]
                x_train = my_encoder.fit(__set_x_data_dict__(), len(y_train))
                x_test = my_encoder.fit(__set_x_data_dict__(is_test=True), len(y_test))
                self.vector_list.append(__init_vector_dict__())

        # for i in range(NUM_FOLDS):
        #     print(self.vector_list[i]["x_test"])

        del self.my_data

    def dump(self, do_show=True):

        def __counting_mortality__(_data):
            count = 0
            for _d in _data:
                if _d == [1]:
                    count += 1

            return count

        if USE_W2V:
            append_name = "_w2v_"
        else:
            append_name = "_"

        if IS_CLOSED:
            file_name = DUMP_PATH + DUMP_FILE + append_name + self.file_name + "_closed"
        else:
            file_name = DUMP_PATH + DUMP_FILE + append_name + self.file_name + "_opened_" + str(NUM_FOLDS)

        with open(file_name, 'w') as outfile:
            json.dump(self.vector_list, outfile, indent=4)
            print("\nsuccess make dump file! - file name is", file_name)

        if do_show:
            for i, data in enumerate(self.vector_list):
                print()
                print("\nData Set", i+1)
                print("Train total count -", str(len(self.vector_list[i]["x_train"]["merge"])).rjust(4),
                      "\tmortality count -", str(__counting_mortality__(self.vector_list[i]["y_train"])).rjust(4))
                print("Test  total count -", str(len(self.vector_list[i]["x_test"]["merge"])).rjust(4),
                      "\tmortality count -", str(__counting_mortality__(self.vector_list[i]["y_test"])).rjust(4))
