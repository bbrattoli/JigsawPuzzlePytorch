IMAGENET_FOLD=path_to_ILSVRC2012_img
CHECKPOINTS_FOLD=path_to_output_folder
GPU=0

python JigsawTrain.py ${IMAGENET_FOLD} --checkpoint=${CHECKPOINTS_FOLD} --classes=1000 --lr=0.1 --gpu=${GPU}
