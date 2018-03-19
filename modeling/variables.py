from collections import OrderedDict

default_info_columns = {
    "scalar": {
        "0": ['D', 'E']
    }
}
initial_info_columns = {
    "scalar": {
        "0": ['G', 'I', 'L', 'M', 'N', 'O', 'Q', 'S', 'T'],
        "start_1": ['H', 'J'],
        "start_1#end_1": ['K']
    },
    "class": ['P', 'R']
}
past_history_columns = {
    "scalar": {
        "0": ['U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC'],
    },
    "class": ['AD']
}

blood_count_columns = {
    "scalar": {
        "0": ['AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN']
    }
}

blood_chemistry_columns = {
    "scalar": {
        "0": ['AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN']
    }
}

# scalar_columns = {
#     "0": ['D', 'E', 'G', 'I', 'L', 'M', 'N', 'O', 'Q', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC'],
#     "start_1": ['H', 'J'],
#     # "end_1": [''],
#     "start_1_end_1": ['K']
# }
# class_columns = ['P', 'R', 'AD']

columns_dict = OrderedDict()
# columns_dict["default"] = default_info_columns
# columns_dict["initial"] = initial_info_columns
# columns_dict["history"] = past_history_columns
# columns_dict["b_count"] = blood_count_columns
columns_dict["b_chemistry"] = blood_chemistry_columns


DUMP_PATH = "modeling/"
DUMP_FILE = "vectors"
