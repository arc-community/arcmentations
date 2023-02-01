"""Noise augmentation classes"""
import functools
import random
from typing import Tuple, Union

import numpy as np

from .helpers import same_aug_for_all_pairs_helper
from ..functional.noise import noiseInput
from arc.interface import BoardPair, Riddle


class Noise(object):
    def __init__(
        self,
        p: float,
        noise_floor: float = 0.05,
        noise_ceiling: float = 0.2,
        kernel_width_max: int = 3,
        kernel_height_max: int = 3,
    ):
        self.p = p
        self.noise_floor = noise_floor
        self.noise_ceiling = noise_ceiling
        self.kernel_width_max = kernel_width_max
        self.kernel_height_max = kernel_height_max
        self.same_aug_for_all_pairs = True

    def __call__(
        self, inp: Union[BoardPair, list[BoardPair], Riddle]
    ) -> Union[BoardPair, list[BoardPair]]:
        func = noiseInput
        # Make sure we are getting a riddle as input
        assert isinstance(inp, Riddle), "Please input a Riddle"
        if random.random() < self.p:
            # Find all colors that are not any input or output board
            all_bps = [i for i in inp.train] + [i for i in inp.test]
            unused_colors = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            for bp in all_bps:
                unused_colors -= set(np.unique(bp.input.np).tolist())
                unused_colors -= set(np.unique(bp.output.np).tolist())

            # Pick a random color (TODO allow for multiple color noise) todo add same aug for all pairs functionality
            if len(unused_colors) == 0:
                return inp
            color = random.choice(list(unused_colors))

            noise_level = random.uniform(self.noise_floor, self.noise_ceiling)
            kernel_size = random.randint(1, self.kernel_width_max)
            kernel_size2 = random.randint(1, self.kernel_height_max)
            noise_size = (kernel_size, kernel_size2)
            param_list = [noise_level, noise_size, color]
            return same_aug_for_all_pairs_helper(
                inp,
                transformation_function=func,
                params_in=param_list,
                same_aug_for_all_pairs=self.same_aug_for_all_pairs,
            )
        else:
            return inp

    def __repr__(self):
        return self.__class__.__name__
