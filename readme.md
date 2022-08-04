

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

```

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
 - Add border to riddles
 - Super resolution (2x or 3x resolution increase)
 - Taurus translate (wraps back around)
 - Mask augmentation (random pixels masked with a sparsity parameter)
 - Unique mapping between color and pattern
 - Static noise augmentation (add static noise)
 - Static noise augmentation that tries to add static with colors not in the puzzle 
 - Some  augmentation that demonstrates the concept of correspondence?
 - Object detector in input and output and finding objects that are in both, then maybe changing the entire object 
 - Repeat board with augmentation such as reflect
 - Join different boards from same riddle into single board (with and without added augmentations)

### Add notes or metadata if the augmentation is lossy
### Generate an inverse transforms for some augmentations so that they can be utilized for Test Time augmentation
