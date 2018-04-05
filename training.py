# -*- coding: utf-8 -*-
import sys
import json
from os import path

try:
    import DeepPredict
except ImportError:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from DeepPredict.dataset.variables import DATA_FILE
from DeepPredict.modeling.variables import DUMP_PATH, DUMP_FILE, TENSOR_PATH
from DeepPredict.learning.train import MyTrain
import DeepPredict.arguments as op


if __name__ == '__main__':
    csv_name = DATA_FILE.split('.')[0]

    if op.FILE_VECTOR:
        file_name = op.FILE_VECTOR
    else:
        if op.USE_W2V:
            append_name = "w2v_"
        else:
            append_name = ""

        if op.USE_ID:
            append_name += op.USE_ID

        if op.IS_CLOSED:
            append_name += "closed_"

        file_name = DUMP_FILE + "_" + append_name + csv_name + "_" + str(op.NUM_FOLDS)

    try:
        with open(DUMP_PATH + file_name, 'r') as file:
            vector_list = json.load(file)
    except FileNotFoundError:
        print("\nPlease execute encoding script !")
        print("make sure whether vector file is existed in", DUMP_PATH, "directory")
    else:
        print("\nRead vectors -", file_name)
        op.show_options()

        train = MyTrain(vector_list)
        train.training(tensor_save=TENSOR_PATH + file_name)
