# SfMLearner
This codebase (in progress) implements the system described in the paper:

Unsupervised Learning of Depth and Ego-Motion from Video

[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Matthew Brown](http://matthewalunbrown.com/research/research.html), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), [David G. Lowe](http://www.cs.ubc.ca/~lowe/home.html)

In CVPR 2017 (**Oral**).

See the [project webpage](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/) for more details. Please contact Tinghui Zhou (tinghuiz@berkeley.edu) if you have any questions.

## Prerequisites
This codebase was developed and tested with Tensorflow 1.0, CUDA 8.0 and Ubuntu 16.04.

## Running the single-view depth demo
We provide the demo code for running our single-view depth prediction model. First, download the pre-trained model by running the following
```bash
bash ./models/download_model.sh
```
Then you can use the provided ipython-notebook `demo.ipynb` to run the demo.

## Prepare data for training
In order to train the model using the provided code, the data needs to be formatted in certain manner. 

For the [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) dataset, run the following command
```bash
python data/prepare_train_data.py --dataset_dir=/path/to/raw/kitti/dataset/ --dataset_name='kitti_raw_eigen' --dump_root=/path/to/resulting/formatted/data/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4
```

For the [Cityscapes](https://www.cityscapes-dataset.com/) dataset, run the following command
```bash
python data/prepare_train_data.py --dataset_dir=/path/to/cityscapes/dataset/ --dataset_name='cityscapes' --dump_root=/path/to/resulting/formatted/data/ --seq_length=3 --img_width=416 --img_height=171 --num_threads=4
```
Notice that for Cityscapes the `img_height` is set to 171 because we crop out the bottom part of the image that contains the car logo, and the resulting image still has height 128.

## TODO List (after NIPS deadline)
- Full training code for Cityscapes and KITTI.
- Evaluation code for the KITTI experiments.

## Disclaimer
This is the authors' implementation of the system described in the paper and not an official Google product.