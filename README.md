# JigsawPuzzlePytorch
Pytorch implementation of the paper ["Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles"](https://arxiv.org/abs/1603.09246) by Mehdi Noroozi [GitHub](https://github.com/MehdiNoroozi/JigsawPuzzleSolver)

**Partially tested**
**Performances Coming Soon**

# Dependencies
- Tested with Python 2.7
- [Pytorch](http://pytorch.org/) v0.3
- [Tensorflow](https://www.tensorflow.org/) is used for logging. 
  Remove the Logger all scripts if tensorflow is missing

# Train the JigsawPuzzleSolver
## Setup Loader
Two DataLoader are provided:
- ImageLoader: per each iteration it loads data in image format (jpg,png ,...)
    - *Dataset/JigsawImageLoader.py* uses PyTorch DataLoader and iterator
    - *Dataset/ImageDataLoader.py* custom implementation.

The default loader is *JigsawImageLoader.py*. *ImageDataLoader.py* is slightly faster when using single core.

The images can be preprocessed using *_produce_small_data.py_* which resize the image to 256, keeping the aspect ratio, and crops a patch of size 255x255 in the center.

## Run Training
Fill the path information in *run_jigsaw_training.sh*. 
IMAGENET_FOLD needs to point to the folder containing *ILSVRC2012_img_train*.

```
./run_jigsaw_training.sh [GPU_ID]
```
or call the python script
```
python JigsawTrain.py [*path_to_imagenet*] --checkpoint [*path_checkpoints_and_logs*] --gpu [*GPU_ID*] --batch [*batch_size*]
```
By default the network uses 1000 permutations with maximum hamming distance selected using *select_permutations.py*.

To change the file name loaded for the permutations, open the file *JigsawLoader.py* and change the permutation file in the method *retrive_permutations*

# Details:
- The input of the network should be 64x64, but I need to resize to 75x75,
  otherwise the output of conv5 is 2x2 instead of 3x3 like the official architecture
- Jigsaw trained using the approach of the paper: SGD, LRN layers, 70 epochs
- Implemented *shortcuts*: spatial jittering, normalize each patch indipendently, color jittering, 30% black&white image
- The LRN layer crushes with a PyTorch version older than 0.3

# ToDo
- TensorboardX
- LMDB DataLoader
