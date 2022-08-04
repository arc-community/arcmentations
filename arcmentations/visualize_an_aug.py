import functools
import random
import time
from arc.utils.dataset import get_riddles
from arcmentations import functional
from arcmentations.augmentations.color import RandomColor
from arcmentations.augmentations.helpers import same_aug_for_all_pairs_helper
from arcmentations.augmentations.spatial import DirectionTypeRandomDouble, RandomDoubleInputBoard, RandomReflect, RandomRotate
from arcmentations.vis_helpers import plot_task
from arcmentations.augmentations import RandomCropInputAndOuput
from torchvision import transforms
if __name__ == "__main__":
    train_riddles = get_riddles(["training"])
    riddle = train_riddles[1]
    transform = transforms.Compose([
        transforms.RandomOrder([
            RandomCropInputAndOuput(1, same_aug_for_all_pairs=True),
            RandomDoubleInputBoard(1, same_aug_for_all_pairs=True),
        ]),
	    RandomColor(1, same_aug_for_all_pairs=True,include_0=False),
        RandomRotate(0.5, True),
        RandomReflect(0.5, True),
    ])
    riddle.train = transform(riddle.train)
    
    plot_task(riddle)
    time.sleep(100000)

