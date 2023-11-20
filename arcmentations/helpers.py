from arc.interface import Riddle
import json
from arcmentations.augmentations.helpers import same_aug_for_all_pairs_helper

def json_string_inout_wrapper_riddle(func):

    def wrapper(*args, **kwargs):
        if not isinstance(args[0], str):
            params = args[1:]
            return same_aug_for_all_pairs_helper(args[0], func, True, params_in= params, kwargs=kwargs)
        else:
            first_arg = args[0]
            first_arg = json.loads(first_arg)
            riddle_obj = Riddle(**first_arg)
            params = args[1:]
            ret = same_aug_for_all_pairs_helper(riddle_obj, func, True, params_in=params, kwargs=kwargs)
            return ret.json()
    return wrapper

# if needed can make json_string_inout_wrapper_boardpair
