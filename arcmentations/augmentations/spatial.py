import random
from arc.interface import BoardPair, Riddle
from typing import Union
from .. import functional
from ..functional import Direction

from arcmentations.augmentations.helpers import p_and_same_aug_helper, same_aug_for_all_pairs_helper
class RandomCropInputAndOuput:
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

    def __call__(self, input:Union[BoardPair,list[BoardPair], Riddle])->Union[BoardPair,list[BoardPair], Riddle]:
        params = dict(cols_to_crop=self.cols_to_crop, rows_to_crop=self.rows_to_crop)
        func = functional.cropInputAndOutput
        return p_and_same_aug_helper(input, func, self.p,self.same_aug_for_all_pairs,self.get_params, **params)

class RandomTaurusTranslate:
    def __init__(self, p:float, same_aug_for_all_pairs:bool,max_x_translate:int,max_y_translate:int):
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.max_x = max_x_translate
        self.max_y = max_y_translate
        assert max_x_translate >= 1, "max_x_translate must be >= 1"
        assert max_y_translate >= 1, "max_y_translate must be >= 1"
        
    @staticmethod
    def get_params(seed, **kwargs):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)
        x_choice = random.choice(range(0,kwargs['max_x']+1))
        y_choice = random.choice(range(0,kwargs['max_y']+1))
        dir_choice_x = random.choice([-1,1])
        dir_choice_y = random.choice([-1,1])
        return x_choice,y_choice,dir_choice_x,dir_choice_y

    def __call__(self, input:Union[BoardPair,list[BoardPair], Riddle])->Union[BoardPair,list[BoardPair], Riddle]:
        params = dict(max_x=self.max_x,max_y=self.max_y)
        func = functional.taurusTranslate
        return p_and_same_aug_helper(input, func, self.p,self.same_aug_for_all_pairs, self.get_params, **params)


class RandomCropInputOnly:
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

    def __call__(self, input:Union[BoardPair,list[BoardPair], Riddle])->Union[BoardPair,list[BoardPair], Riddle]:
        params = dict(cols_to_crop=self.cols_to_crop, rows_to_crop=self.rows_to_crop)
        func = functional.cropInputOnly
        return p_and_same_aug_helper(input, func, self.p,self.same_aug_for_all_pairs,self.get_params, **params)

class RandomFloatRotate:
    def __init__(self, p:float, same_aug_for_all_pairs:bool, max_degree_delta:int=180):
        """
        Randomly rotates boards by any number of degrees then re-rasterizes into a grid 
        """
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.max_degree_delta = max_degree_delta
        assert max_degree_delta >= 1, "max_degree_delta must be >= 1"
        assert max_degree_delta <= 180, "max_degree_delta must be <= 180"
        
    @staticmethod
    def get_params(seed, max_degree_delta):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)
        r = random.randint(-max_degree_delta,max_degree_delta)
        return (r,)

    def __call__(self, input:Union[BoardPair,list[BoardPair],Riddle])->Union[BoardPair,list[BoardPair],Riddle]:
        params = dict(max_degree_delta=self.max_degree_delta)
        func = functional.floatRotateAll
        return p_and_same_aug_helper(input, func, self.p,self.same_aug_for_all_pairs,self.get_params, **params)        
class RandomDoubleInputBoard:
    def __init__(self, p:float, same_aug_for_all_pairs:bool,possible_separations:list[int]=[-1,1,2], direction_type:Direction=Direction.both, random_z_index:bool=True):
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
        if direction_type == Direction.both:
            direction_type = random.choice([Direction.horizontal,Direction.vertical])
        if kwargs['random_z_index']:
            z_index_of_original = bool(random.choice([0,1]))
        else:
            z_index_of_original = False
        is_horizontal = direction_type == Direction.horizontal

        return sep, is_horizontal, z_index_of_original

    def __call__(self, input:Union[BoardPair,list[BoardPair],Riddle])->Union[BoardPair,list[BoardPair],Riddle]:
        params = dict(possible_separations=self.possible_separations,direction_type=self.direction_type,random_z_index=self.random_z_index)
        func = functional.doubleInputBoard
        return p_and_same_aug_helper(input, func, self.p,self.same_aug_for_all_pairs,self.get_params, **params)    
    
class RandomRotate:
    def __init__(self, p:float, same_aug_for_all_pairs:bool, number_of_90_degrees_rotations:int=3):
        """
        Randomly rotates boards by multiples of 90 degree increments
        """
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.number_of_90_degrees_rotations = number_of_90_degrees_rotations
        
    @staticmethod
    def get_params(seed, number_of_90_degrees_rotations):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)
        r = number_of_90_degrees_rotations
        assert r >= 1 and r <=3, f"number_of_90_degrees_rotations must be between 1 and 3, not {r}"
        rotation_to_use = random.choice(list(range(r+1)))
        return rotation_to_use

    def __call__(self, input:Union[BoardPair,list[BoardPair],Riddle])->Union[BoardPair,list[BoardPair],Riddle]:
        params = dict(number_of_90_degrees_rotations=self.number_of_90_degrees_rotations)
        func = functional.rotate
        return p_and_same_aug_helper(input, func, self.p,self.same_aug_for_all_pairs,self.get_params, **params)        
        
class RandomReflect:
    def __init__(self, p:float, same_aug_for_all_pairs:bool, x_axis:bool=True, y_axis:bool=True):
        """
        Randomly reflected boards across axes
        """
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.x_axis = x_axis
        self.y_axis = y_axis
        
    @staticmethod
    def get_params(seed, x_axis:bool, y_axis:bool):
        """
        Get parameters for this augmenter.
        """
        random.seed(seed)
        if x_axis:
            x_axis = random.choice([True, False])
        if y_axis:
            y_axis = random.choice([True, False])
        return x_axis, y_axis

    def __call__(self, input:Union[BoardPair,list[BoardPair],Riddle])->Union[BoardPair,list[BoardPair],Riddle]:
        params = dict(x_axis=self.x_axis, y_axis=self.y_axis)
        func = functional.reflect
        return p_and_same_aug_helper(input, func, self.p, self.same_aug_for_all_pairs,self.get_params, **params)
class RandomPad:
    def __init__(self, p:float, same_aug_for_all_pairs:bool, pad_sizes:list[int]=[1,2], pad_values:list[int]=[0]):
        """
        Randomly pads boards with a given value
        """
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.pad_sizes = pad_sizes
        self.pad_values = pad_values
        
    @staticmethod
    def get_params(seed, pad_sizes, pad_values):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)
        s = random.choice(pad_sizes)
        v = random.choice(pad_values)
        return s,v

    def __call__(self, input:Union[BoardPair,list[BoardPair],Riddle])->Union[BoardPair,list[BoardPair],Riddle]:
        params = dict(pad_sizes=self.pad_sizes, pad_values=self.pad_values)
        func = functional.padInputOutput
        return p_and_same_aug_helper(input, func, self.p, self.same_aug_for_all_pairs,self.get_params, **params)
class RandomPadInputOnly:
    def __init__(self, p:float, same_aug_for_all_pairs:bool, pad_sizes:list[int]=[1,2], pad_values:list[int]=[0]):
        """
        Randomly pads boards with a given value
        """
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.pad_sizes = pad_sizes
        self.pad_values = pad_values
        
    @staticmethod
    def get_params(seed, pad_sizes, pad_values):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)
        s = random.choice(pad_sizes)
        v = random.choice(pad_values)
        return s,v

    def __call__(self, input:Union[BoardPair,list[BoardPair],Riddle])->Union[BoardPair,list[BoardPair],Riddle]:
        params = dict(pad_sizes=self.pad_sizes, pad_values=self.pad_values)
        func = functional.padInputOnly
        return p_and_same_aug_helper(input, func, self.p, self.same_aug_for_all_pairs,self.get_params, **params)


class RandomSuperResolution:
    def __init__(self, p:float, same_aug_for_all_pairs:bool, super_resolution_factors:list[int]=[2,3], stretch_axis:list[Direction] = [Direction.both]):
        """
        Randomly explodes boards by a given factor
        """
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs
        self.super_resolution_factors = super_resolution_factors
        self.stretch_axis = stretch_axis
        for s in super_resolution_factors:
            assert s >= 1, f"super_resolution_factors must be >= 1, not {s}"
    @staticmethod
    def get_params(seed, super_resolution_factors, stretch_axis):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)   
        s = random.choice(super_resolution_factors)
        axes = random.choice(stretch_axis)
        return s, axes
    def __call__(self, input:Union[BoardPair,list[BoardPair],Riddle])->Union[BoardPair,list[BoardPair],Riddle]:
        params = dict(super_resolution_factors=self.super_resolution_factors, stretch_axis=self.stretch_axis)
        func = functional.superResolution
        return p_and_same_aug_helper(input, func, self.p, self.same_aug_for_all_pairs, self.get_params, **params)
    

class Shuffle:
    def __init__(
        self,
        p: float,
        same_aug_for_all_pairs: bool = True,
    ):
        self.p = p
        self.same_aug_for_all_pairs = same_aug_for_all_pairs

    @staticmethod
    def get_params(seed, **kwargs):
        """
        Get parameters for this augmenter. Must use the seed provided
        """
        random.seed(seed)
        num_training = kwargs["num_training"]
        num_test = kwargs["num_test"]
        return random.shuffle(list(range(num_training))), random.shuffle(list(range(num_test)))
    
    def __call__(self, input:Riddle)->Riddle:
        assert isinstance(input, Riddle), "Shuffle only works on Riddles"# todo add pairs
        params = dict(num_training=len(input.train), num_test=len(input.test))
        shuffled_train_list, shuffled_test_list = self.get_params(**params)
        input.train = [input.train[i] for i in shuffled_train_list]
        input.test = [input.test[i] for i in shuffled_test_list]
        return input