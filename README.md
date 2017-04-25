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
Then you can use the provided ipython-notebook `demo.ipynb' to run the demo.

## TODO List
- Full training code for Cityscapes and KITTI.
- Evaluation code for the KITTI experiments.

## Disclaimer
This is the authors' implementation of the system described in the paper. This is not an official Google product.