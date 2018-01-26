# JigsawPuzzlePytorch
Pytorch implementation of the paper ["Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles"](https://arxiv.org/abs/1603.09246) [Caffe Implementation](https://github.com/MehdiNoroozi/JigsawPuzzleSolver) by Mehdi Noroozi

**Still not fully tested**

# Dependencies
- Tested with Python 2.7
- [Pytorch](http://pytorch.org/) v0.3
- [Tensorflow](https://www.tensorflow.org/) is used for logging. 
  Remove the Logger all scripts if tensorflow is missing

# Train the JiggsawPuzzleSolver
Fill the path information in **_run_jigsaw_training.sh_** and run
```
./run_jigsaw_training.sh [GPU_ID]
```
or call the python script
```
python JigsawTrain.py [path_to_training_set_folder] --checkpoint [path_where_to_store_checkpoints_and_logs] --gpu [GPU_ID] --batch [batch_size]
```
By default the network uses 1000 permutations with maximum hamming distance selected using **_select_permutations.py_**
To change the file loaded for the permutations, open the file **_JigsawLoader.py_** and change the permutation file in the method **_retrive_permutations_**

# Details:
- The input of the network should be 64x64, but I need to resize to 75x75,
  otherwise the output of conv5 is 2x2 instead of 3x3 like the weights from
  the official website
- Jigsaw trained using the approach of the paper: SGD, LRN layers
- Implemented *shortcuts*: teil spatial random jittering, normalize each patch indipendently, color jitter
- The LRN layer crushes with a PyTorch version older than 0.3

# ToDo
- TensorboardX
