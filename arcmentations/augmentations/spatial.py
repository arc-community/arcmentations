import functools
import random
from arc.interface import BoardPair
from typing import Union
from .. import functional

from arcmentations.augmentations.helpers import same_aug_for_all_pairs_helper

class SpatialBaseClass(object):
    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_params(seed, **kwargs):
        raise NotImplementedError
        
    def _call(self, input, func, **params):
        if random.random() < self.p:
            get_params_method = functools.partial(self.get_params, **params)
            return same_aug_for_all_pairs_helper(input, func, self.same_aug_for_all_pairs, get_params_method)
        else:
            return input

    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__
    
    

class RandomCropInputAndOuput(SpatialBaseClass):
    def __init__(self, p:float, same_aug_for_all_pairs:bool,possible_num_cols_to_crop:list[int]=[1,2],possible_num_rows_to_crop:list[int]=[1,2]):
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.cols_to_crop = possible_num_cols_to_crop
        self.rows_to_crop = possible_num_rows_to_crop
        
    @staticmethod
    def get_params(seed, **kwargs):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)
        col_choice = random.choice(kwargs['cols_to_crop'])
        row_choice = random.choice(kwargs['rows_to_crop'])
        dir_choice_col = random.choice([-1,1])
        dir_choice_row = random.choice([-1,1])
        return col_choice,row_choice,dir_choice_col,dir_choice_row

    def __call__(self, input:Union[BoardPair,list[BoardPair]])->Union[BoardPair,list[BoardPair]]:
        params = dict(cols_to_crop=self.cols_to_crop, rows_to_crop=self.rows_to_crop)
        return self._call(input, functional.cropInputAndOutput, **params)


from enum import Enum
class DirectionTypeRandomDouble(Enum):
    horizontal = 1
    vertical = 2
    both = 3
    
    
class RandomDoubleInputBoard(SpatialBaseClass):
    def __init__(self, p:float, same_aug_for_all_pairs:bool,possible_separations:list[int]=[-1,1,2], direction_type:DirectionTypeRandomDouble=DirectionTypeRandomDouble.both, random_z_index:bool=True):
        """
        Doubles the input board only, leaving the output board the same.
        possible_separations: list of possible separations between the original and copy.
        separation can be negative meaning some overlap between the two boards will take place.
        direction_type: the direction of concatnation.
        random_z_index: whether to randomize the z index of the overlap. (i.e randomly choose weather the left board might be on top of the right board or the other way around)
        """
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.possible_separations = possible_separations
        self.direction_type = direction_type
        self.random_z_index = random_z_index

    @staticmethod
    def get_params(seed, **kwargs):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)
        sep = random.choice(kwargs['possible_separations'])
        direction_type = kwargs['direction_type']
        if direction_type == DirectionTypeRandomDouble.both:
            direction_type = random.choice([DirectionTypeRandomDouble.horizontal,DirectionTypeRandomDouble.vertical])
        if kwargs['random_z_index']:
            z_index_of_original = bool(random.choice([0,1]))
        else:
            z_index_of_original = False
        is_horizontal = direction_type == DirectionTypeRandomDouble.horizontal

        return sep, is_horizontal, z_index_of_original

    def __call__(self, input:Union[BoardPair,list[BoardPair]])->Union[BoardPair,list[BoardPair]]:
        params = dict(possible_separations=self.possible_separations,direction_type=self.direction_type,random_z_index=self.random_z_index)
        func = functional.doubleInputBoard
        return self._call(input, func, **params)
    
    
class RandomRotate(SpatialBaseClass):
    def __init__(self, p:float, same_aug_for_all_pairs:bool, number_of_90_degrees_rotations:int=3):
        """
        Randomly rotates boards by multiples of 90 degree increments
        """
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.number_of_90_degrees_rotations = number_of_90_degrees_rotations
        
    @staticmethod
    def get_params(seed, number_of_90_degrees_rotations:int=-1):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)
        r = number_of_90_degrees_rotations
        assert r >= -1 and r <=3, f"number_of_90_degrees_rotations must be between -1 and 3, not {r}"
        if r==-1: r = random.randint(0,4)
        return r

    def __call__(self, input:Union[BoardPair,list[BoardPair]])->Union[BoardPair,list[BoardPair]]:
        params = dict(number_of_90_degrees_rotations=self.number_of_90_degrees_rotations)
        func = functional.rotate
        return self._call(input, func, **params)
        
        
class RandomReflect(SpatialBaseClass):
    def __init__(self, p:float, same_aug_for_all_pairs:bool, x_axis:bool=True, y_axis:bool=True):
        """
        Randomly reflected boards across axes
        """
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.x_axis = x_axis
        self.y_axis = y_axis
        
    @staticmethod
    def get_params(seed, x_axis:bool=None, y_axis:bool=None):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)
        if x_axis is None:
            x_axis = random.choice([True, False])
        if y_axis is None:
            y_axis = random.choice([True, False])
        return x_axis, y_axis

    def __call__(self, input:Union[BoardPair,list[BoardPair]])->Union[BoardPair,list[BoardPair]]:
        params = dict(x_axis=self.x_axis, y_axis=self.y_axis)
        func = functional.reflect
        return self._call(input, func, **params)