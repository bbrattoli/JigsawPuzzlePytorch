# JigsawPuzzlePytorch
Pytorch implementation of the paper ["Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles"](https://arxiv.org/abs/1603.09246) by Mehdi Noroozi [GitHub](https://github.com/MehdiNoroozi/JigsawPuzzleSolver)

**Partially tested**
**Performances Coming Soon**

# Dependencies
- Tested with Python 2.7
- [Pytorch](http://pytorch.org/) v0.3
- [Tensorflow](https://www.tensorflow.org/) is used for logging. 
  Remove the Logger all scripts if tensorflow is missing

# Train the JiggsawPuzzleSolver
## Setup Loader
Two DataLoader are provided:
- **_Dataset/JigsawImageLoader.py_**: per each iteration it loads data in image format (jpg,png ,...)
- **_Dataset/JigsawTilesLoader.py_**: load all pre-processed images into memory. Preprocessed data need to be produced using **_Dataset/produce_jigsaw_data.py_**
The default loader is **_JigsawImageLoader.py_**. **_JigsawTilesLoader.py_** can be used with the flage **_--processed_**

To start training, the DataLoader needs the path to the Imagenet folder containing **_ILSVRC2012_img_train_**. 
Fill the path information in **_run_jigsaw_training.sh_**

## Run Training
```
./run_jigsaw_training.sh [GPU_ID]
```
or call the python script
```
python JigsawTrain.py [path_to_imagenet] --checkpoint [path_checkpoints_and_logs] --gpu [GPU_ID] --batch [batch_size]
```
By default the network uses 1000 permutations with maximum hamming distance selected using **_Utils/select_permutations.py_**
To change the file name loaded for the permutations, open the file **_JigsawLoader.py_** and change the permutation file in the method **_retrive_permutations_**

# Details:
- The input of the network should be 64x64, but I need to resize to 75x75,
  otherwise the output of conv5 is 2x2 instead of 3x3 like the official architecture
- Jigsaw trained using the approach of the paper: SGD, LRN layers
- Implemented *shortcuts*: teil spatial random jittering, normalize each patch indipendently, color jitter
- The LRN layer crushes with a PyTorch version older than 0.3

# ToDo
- TensorboardX
- LMDB DataLoader
