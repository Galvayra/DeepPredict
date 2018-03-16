from .myOneHotEncoder import MyOneHotEncoder
from .variables import NUM_FOLDS, IS_CLOSED


class MyVector:
    def __init__(self, my_data):
        def __init_vector_list__():
            vector_dict = {
                "x_train": list(),
                "y_train": list(),
                "x_test": list(),
                "y_test": list()
            }

            if IS_CLOSED:
                return [vector_dict]
            else:
                return [vector_dict for i in range(NUM_FOLDS)]

        self.my_data = my_data
        self.vector_list = __init_vector_list__()
        self.__set_vector_list__()

    def __set_vector_list__(self):


        num_folds = len(self.vector_list)
        my_encoder = MyOneHotEncoder()



        # if IS_CLOSED:
        #     pass
        # else:
        #     subset_size = int(len(self.my_data.y_data) / NUM_FOLDS) + 1
        #
        #     for i in range(NUM_FOLDS):
        #
        #         y_train = self.my_data.y_data[:i * subset_size] + self.my_data.y_data[(i + 1) * subset_size:]
        #         y_test = self.my_data.y_data[i * subset_size:][:subset_size]
