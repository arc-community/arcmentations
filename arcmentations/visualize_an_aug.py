import copy
import functools
import random
import time
from arc.utils.dataset import get_riddles
from arcmentations import functional
import arcmentations
from arcmentations.augmentations.color import RandomColor
from arcmentations.augmentations.helpers import same_aug_for_all_pairs_helper
from arcmentations.augmentations.spatial import Direction, RandomDoubleInputBoard, RandomFloatRotate, RandomPadInputOnly, RandomPad, RandomReflect, RandomRotate, RandomSuperResolution, RandomTaurusTranslate\
    , RandomFloatRotate2 , RandomQuasiRotate,RandomFlipInputAndOutput
from arcmentations.vis_helpers import plot_task, plot_pairs
from arcmentations.augmentations import RandomCropInputAndOuput
from arcmentations.helpers import json_string_inout_wrapper_riddle

#   from torchvision import transforms
if __name__ == "__main__":
    train_riddles = get_riddles(["training"])
    for riddle in train_riddles[10:16]:
        # transform = transforms.Compose([
        #     # transforms.RandomOrder([
        #     #     RandomCropInputAndOuput(1, same_aug_for_all_pairs=True),
        #     #     # RandomDoubleInputBoard(1, same_aug_for_all_pairs=True),
        #     # ]),
        #     # RandomColor(1, same_aug_for_all_pairs=True, include_0=False),
        #     # RandomRotate(1, True),

        #     RandomReflect(1, same_aug_for_all_pairs=False),
        #     RandomSuperResolution(1, same_aug_for_all_pairs=False, stretch_axis=[Direction.both,Direction.horizontal,Direction.vertical]),

        #     transforms.RandomChoice([
        #         transforms.Compose([
        #             RandomTaurusTranslate(0.8, same_aug_for_all_pairs=False, max_x_translate=100, max_y_translate=100),
        #             RandomPad(1, same_aug_for_all_pairs=False, pad_sizes=[1], pad_values=[1,2,3,4,5,6,7,8,9]),
        #             RandomFloatRotate(1, same_aug_for_all_pairs=False,max_degree_delta = 180),
        #             RandomPad(1, same_aug_for_all_pairs=False, pad_sizes=[0,1,2], pad_values=[0]),
        #         ]),
        #         transforms.Compose([
        #             RandomPad(1, same_aug_for_all_pairs=False, pad_sizes=[0,1,2,3,4,5], pad_values=[0]),
        #             RandomTaurusTranslate(1, same_aug_for_all_pairs=False, max_x_translate=100, max_y_translate=100),
        #             RandomPad(0.8, same_aug_for_all_pairs=False, pad_sizes=[1], pad_values=[0,1,2,3,4,5,6,7,8,9]),
        #         ]),
        #     ]),

        # ])

        transform = RandomFlipInputAndOutput(1)
        #transform = RandomReflect(1)
        riddle_transformed = transform(copy.deepcopy(riddle))
        path_to_save = './tmp/time_{}.png'.format(time.time())
        plot_task(riddle_transformed, save=False,save_train=False,path_to_save=path_to_save)
        plot_task(riddle_transformed, save=True, save_train=True, path_to_save=path_to_save)
        plot_task(riddle, save=True, path_to_save='./tmp/train.png',save_train=True)
        #time.sleep(100000)
    # quasi_rotate_fun = arcmentations.augmentations.functional.quasiRotate
    # quasi_rotate_json_fun = json_string_inout_wrapper_riddle(quasi_rotate_fun)
    # riddle = copy.deepcopy(train_riddles[6].json())
    # riddle_transformed = quasi_rotate_json_fun(
    #     riddle,
    #     angleHor=25,
    #     angleVer=0,
    #     startTopHor=True,
    #     startLeftVer=True,
    #     doHorFirst=True
    # )
    # print(riddle_transformed)
    # a = 1

