import functools
import random
import time
from arc.utils.dataset import get_riddles
from arcmentations import functional
from arcmentations.augmentations.color import RandomColor
from arcmentations.augmentations.helpers import same_aug_for_all_pairs_helper
from arcmentations.augmentations.spatial import DirectionTypeRandomDouble, RandomDoubleInputBoard
from arcmentations.vis_helpers import plot_task
from arcmentations.augmentations import RandomCropInputAndOuput

if __name__ == "__main__":
    train_riddles = get_riddles(["training"])
    riddle = train_riddles[1]
    # riddle.train = RandomCropInputAndOuput(p=1, same_aug_for_all_pairs=True, possible_num_cols_to_crop=[1], possible_num_rows_to_crop=[0])(riddle.train)
    plot_task(riddle)
    params = RandomColor.get_params(seed=random.random(), include_0=False)

    riddle.train = same_aug_for_all_pairs_helper(riddle.train, functional.permute_color, True, params_in=params)
    
    plot_task(riddle)
    time.sleep(100000)

