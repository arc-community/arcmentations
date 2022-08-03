import random
from typing import Union
from arc.interface import BoardPair

def same_aug_for_all_pairs_helper(input:Union[BoardPair,list[BoardPair]], transformation_function,same_aug_for_all_pairs:bool,get_params_method=None,params_in=None):
    if type(input) == list:
        if same_aug_for_all_pairs:
            assert params_in is not None or get_params_method is not None, "params_in or get_params_method must be provided"
            params = get_params_method( seed=random.random()) if get_params_method is not None else params_in
            output = [transformation_function(input_pair,*params) for input_pair in input]
            return output
        else:
            assert get_params_method is not None, "get_params_method must be provided if same_aug_for_all_pairs is False"
            output = []
            for i in range(len(input.pairs)):
                params = get_params_method(seed=random.random())
                output.append(transformation_function(input[i],*params))
            return output
    elif type(input) == BoardPair:
            assert params_in is not None or get_params_method is not None, "params_in or get_params_method must be provided"
            params = get_params_method(seed=random.random()) if params_in is None else params_in
            return transformation_function(input,*params)
    else:
        raise Exception("Input must be a BoardPair or a list of BoardPairs")
