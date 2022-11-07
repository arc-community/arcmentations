

requires https://github.com/arc-community/arc installed

is interoperable with torchvision transforms so you can do things like:

```
from torchvision import transforms

transform = transforms.compose([
	transforms.RandomOrder([
        RandomCropInputAndOuput(0.5, same_aug_for_all_pairs=True),
        RandomDoubleInputBoard(0.5, same_aug_for_all_pairs=True),
    ]),
	RandomColor(0.5, same_aug_for_all_pairs=True)
])

aug_training_riddles = transform(training_riddles)

```
See the demonstrations notebook for more details.

Each augmentation class has a corresponding function in arcmentations.functional that take in params to do a specific non random transformation. which is also useful for people that just want to use the same augmentation for all their data.


you can sample params for a specific function in arcmentations.functional by calling the Random* class's static method `get_params()`

for example:

```
from arcmentations.augmentations import RandomCropInputAndOuput

params = RandomCropInputAndOuput.get_params(*args, **kwargs)
for riddle in riddles:
    riddle = functional.cropInputAndOutput(riddle, *params)

```
## Todo Items
### Augmentations to add:
 - Add padding to riddles
 - Octagonal grid rotations, much more determinisic and accurate than the current RandomFloatRotate
 - Super resolution (2x or 3x resolution increase)
 - Torus translate (wraps back around) (can be combined with pad augmentation to make it more interesting)
 - Mask augmentation (random pixels masked with a sparsity parameter)
 - Unique mapping between color and pattern
 - Static noise augmentation (add static noise)
 - Static noise augmentation that tries to add static with colors not in the puzzle 
 - Reorder puzzle training boards (randomly)
 - change the test board to one of the training ones and a training board to the test one, ranadomly
 - Some  augmentation that demonstrates the concept of correspondence?
 - Object detector in input and output and finding objects that are in both, then maybe changing the entire object 
 - Repeat board with augmentation such as reflect
 - Join different boards from same riddle into single board (with and without added augmentations)
  - Grow input or output board by 2x horizontally and or vertically by spacing out the original board with a background color or random color (maybe this helps with adding a  correspondance prior?)
  - super res scale everything except background color by a random factor, maybe only across a specifix x/y axis
  - flag to specify input/output/both on a transformation (impl using helpers not base class)
  - flag or helper to artificially generate more board pairs with AND without an augmentation applied. Useful especially in the case where the augmentation is only applied to input board.
  - update demo notebook with full riddle augs 
  

### Add notes or metadata if the augmentation is lossy
### Generate an inverse transforms for some augmentations so that they can be utilized for Test Time Augmentation (TTA)
