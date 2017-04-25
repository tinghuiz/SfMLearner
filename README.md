# SfMLearner
An unsupervised learning framework for depth and ego-motion estimation from monocular videos

## Prerequisites
This codebase was developed and tested with Tensorflow 1.0, CUDA 8.0 and Ubuntu 16.04.

## Running the single-view depth demo
We provide the demo code for running our single-view depth prediction model. First, download the pre-trained model by running the following
```bash
bash ./models/download_model.sh
```
Then you can use the provided ipython-notebook `demo.ipynb' to run the demo.

## TODO List
- Full training code for Cityscapes and KITTI.
- Evaluation code for the KITTI experiments.

## Disclaimer
This is the authors' implementation of the system described in the paper. This is not an official Google product.