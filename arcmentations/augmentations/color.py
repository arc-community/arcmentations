import functools
import random
from arc.interface import BoardPair
from typing import Union
from arcmentations import functional

from arcmentations.augmentations.helpers import same_aug_for_all_pairs_helper

class RandomColor(object):
    def __init__(self, p:float,include_0:bool = True, same_aug_for_all_pairs:bool = True):
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.include_0 = include_0

    @staticmethod
    def get_params( seed, **kwargs):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)
        if kwargs['include_0']:
            colours = random.sample(list(range(10)),10)
        else: 
            colours = [0] + random.sample(list(range(1,10)),9)
        return (colours,)

    def __call__(self, input:Union[BoardPair,list[BoardPair]],params=None)->Union[BoardPair,list[BoardPair]]:
        if random.random() < self.p:
            get_params_method = functools.partial(self.get_params,include_0=self.include_0)
            return same_aug_for_all_pairs_helper(input, get_params_method=get_params_method, same_aug_for_all_pairs = self.same_aug_for_all_pairs, transformation_function=functional.permute_color)
        else:
            return input

    def __repr__(self):
        return self.__class__.__name__