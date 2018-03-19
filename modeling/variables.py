from collections import OrderedDict

default_info_columns = {
    "scalar": {
        "0": ['D']
    },
    "class": ['E']
}

initial_info_columns = {
    "scalar": {
        "0": ['G', 'I', 'L', 'M', 'N', 'O', 'Q'],
        "start_1": ['H', 'J'],
        "start_1#end_1": ['K']
    },
    "class": ['P', 'R', 'S', 'T']
}

past_history_columns = {
    "class": ['U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD']
}

blood_count_columns = {
    "scalar": {
        "0": ['AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN']
    }
}

blood_chemistry_columns = {
    "scalar": {
        "0": ['AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ',
              'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BO', 'BP',
              'BQ', 'BR', 'BS']
    },
    "class": ['BT', 'BU', 'BV']
}

abga_columns = {
    "scalar": {
        "0": ['BW', 'BX', 'BY', 'BZ', 'CA', 'CB', 'CC']
    }
}

culture_columns = {
    "class": ['CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK', 'CL']
}

influenza_columns = {
    "scalar": {
        "0": ['CM', 'CN']
    }
}

ct_columns = {
    "scalar": {
        "0": ['CO', 'CP']
    }
}

columns_dict = OrderedDict()
# columns_dict["default"] = default_info_columns
# # columns_dict["initial"] = initial_info_columns
# columns_dict["history"] = past_history_columns
# # columns_dict["b_count"] = blood_count_columns
# # columns_dict["b_chemistry"] = blood_chemistry_columns
# # columns_dict["abga"] = abga_columns
columns_dict["culture"] = culture_columns
# # columns_dict["influenza"] = influenza_columns
# # columns_dict["ct"] = ct_columns


DUMP_PATH = "modeling/"
DUMP_FILE = "vectors"
