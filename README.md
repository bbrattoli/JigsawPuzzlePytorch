# JigsawPuzzlePytorch
Pytorch implementation of the model from the paper ["Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles"](https://arxiv.org/abs/1603.09246)

**Still not fully tested**

# Dependencies
- Tested with Python 2.7
- [Pytorch](http://pytorch.org/) needs to be installed
- [Tensorflow](https://www.tensorflow.org/) is used for logging. 
  Remove the Logger all scripts if tensorflow is missing

# Train the JiggsawPuzzleSolver
```
python JigsawTrain.py [path_to_training_set_folder] --classes [10, 100 or 1000] --checkpoint [path_where_to_store_checkpoints_and_logs] --gpu [GPU_ID] --lr [LR] --batch [batch_size] --epoch [num_of_epochs]
```
By default the network uses 1000 permutations with maximum hamming distance selected using **_JpsTraininig/select_permutations.py_**
To change the number of permutation, open the file **_JpsTraininig/JigsawLoader.py_** and change the permutation file in the method *__retrive_permutations*

# Train Imagenet classifier
```
python Imagenet_finetuning/ImagenetTrain.py [path_to_training_set_folder] [path_to_train.txt] [path_to_validation_set_folder] [path_to_val.txt] --checkpoint [path_where_to_store_checkpoints_and_logs] [path_to_validation_set_folder] [val.txt] --gpu [GPU_ID]
```

# Train Pascal VOC 2007 multi-classes classifier
```
python Pascal_finetuning/PascalTrain.py [path_to_VOC2007] --checkpoint [path_where_to_store_checkpoints_and_logs] [path_to_validation_set_folder] [val.txt] --gpu [GPU_ID]
```

# Train Pascal VOC 2007 detection using Faster-RCNN **Work in progress**


# Details:
- Jigsaw trained using the approach of the paper: SGD, LRN layers
- Implemented *shortcuts*: teil spatial random jittering, normalize each patch indipendently, color jitter
- The input of the network should be 64x64, but I need to resize to 75x75,
  otherwise the output of conv5 is 2x2 instead of 3x3 like the weights from
  the official website
