# SfMLearner
This codebase implements the system described in the paper:

Unsupervised Learning of Depth and Ego-Motion from Video

[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Matthew Brown](http://matthewalunbrown.com/research/research.html), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), [David G. Lowe](http://www.cs.ubc.ca/~lowe/home.html)

In CVPR 2017 (**Oral**).

See the [project webpage](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/) for more details. Please contact Tinghui Zhou (tinghuiz@berkeley.edu) if you have any questions.

<img src='misc/cityscapes_sample_results.gif' width=320>

## Prerequisites
This codebase was developed and tested with Tensorflow 1.0, CUDA 8.0 and Ubuntu 16.04.

## Running the single-view depth demo
We provide the demo code for running our single-view depth prediction model. First, download the pre-trained model by running the following
```bash
bash ./models/download_depth_model.sh
```
Then you can use the provided ipython-notebook `demo.ipynb` to run the demo.

## Preparing training data
In order to train the model using the provided code, the data needs to be formatted in a certain manner. 

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command
```bash
python data/prepare_train_data.py --dataset_dir=/path/to/raw/kitti/dataset/ --dataset_name='kitti_raw_eigen' --dump_root=/path/to/resulting/formatted/data/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4
```
For the pose experiments, we used the KITTI odometry split, which can be downloaded [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Then you can change `--dataset_name` option to `kitti_odom` when preparing the data.

For [Cityscapes](https://www.cityscapes-dataset.com/), download the following packages: 1) `leftImg8bit_sequence_trainvaltest.zip`, 2) `camera_trainvaltest.zip`. Then run the following command
```bash
python data/prepare_train_data.py --dataset_dir=/path/to/cityscapes/dataset/ --dataset_name='cityscapes' --dump_root=/path/to/resulting/formatted/data/ --seq_length=3 --img_width=416 --img_height=171 --num_threads=4
```
Notice that for Cityscapes the `img_height` is set to 171 because we crop out the bottom part of the image that contains the car logo, and the resulting image will have height 128.

## Training
Once the data are formatted following the above instructions, you should be able to train the model by running the following command
```bash
python train.py --dataset_dir=/path/to/the/formatted/data/ --checkpoint_dir=/where/to/store/checkpoints/ --img_width=416 --img_height=128 --batch_size=4
```
You can then start a `tensorboard` session by
```bash
tensorboard --logdir=/path/to/tensorflow/log/files --port=8888
```
and visualize the training progress by opening [https://localhost:8888](https://localhost:8888) on your browser. If everything is set up properly, you should start seeing reasonable depth prediction after ~100K iterations when training on KITTI. 

### Notes
After adding data augmentation and removing batch normalization (along with some other minor tweaks), we have been able to train depth models better than what was originally reported in the paper even without using additional Cityscapes data or the explainability regularization. The provided pre-trained model was trained on KITTI only with smooth weight set to 0.5, and achieved the following performance on the Eigen test split (Table 1 of the paper):

| Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|---------|--------|-------|-----------|-------|-------|-------|
| 0.183   | 1.595  | 6.709 | 0.270     | 0.734 | 0.902 | 0.959 | 

When trained on 5-frame snippets, the pose model obtains the following performanace on the KITTI odometry split (Table 3 of the paper):

| Seq. 09            | Seq. 10            |
|--------------------|--------------------|
| 0.016 (std. 0.009) | 0.013 (std. 0.009) |

## Evaluation on KITTI

### Depth
We provide evaluation code for the single-view depth experiment on KITTI. First, download our predictions (~140MB) by 
```bash
bash ./kitti_eval/download_kitti_depth_predictions.sh
```
Then run
```bash
python kitti_eval/eval_depth.py --kitti_dir=/path/to/raw/kitti/dataset/ --pred_file=kitti_eval/kitti_eigen_depth_predictions.npy
```
If everything runs properly, you should get the numbers for `Ours(CS+K)` in Table 1 of the paper. To get the numbers for `Ours cap 50m (CS+K)`, set an additional flag `--max_depth=50` when executing the above command.

### Pose
We provide evaluation code for the pose estimation experiment on KITTI. First, download the predictions and ground-truth pose data by running
```bash
bash ./kitti_eval/download_kitti_pose_eval_data.sh
```
Notice that all the predictions and ground-truth are 5-frame snippets with the format of `timestamp tx ty tz qx qy qz qw` consistent with the [TUM evaluation toolkit](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation). Then you could run 
```bash
python kitti_eval/eval_pose.py --gtruth_dir=/directory/of/groundtruth/trajectory/files/ --pred_dir=/directory/of/predicted/trajectory/files/
```
to obtain the results reported in Table 3 of the paper. For instance, to get the results of `Ours` for `Seq. 10` you could run
```bash
python kitti_eval/eval_pose.py --gtruth_dir=kitti_eval/pose_data/ground_truth/10/ --pred_dir=kitti_eval/pose_data/ours_results/10/
```

## KITTI Testing code

### Depth
Once you have model trained, you can obtain the single-view depth predictions on the KITTI eigen test split formatted properly for evaluation by running
```bash
python test_kitti_depth.py --dataset_dir /path/to/raw/kitti/dataset/ --output_dir /path/to/output/directory --ckpt_file /path/to/pre-trained/model/file/
```
Again, a sample model can be downloaded by
```bash
bash ./models/download_depth_model.sh
```

### Pose
We also provide sample testing code for obtaining pose predictions on the KITTI dataset with a pre-trained model. You can obtain the predictions formatted as above for pose evaluation by running
```bash
python test_kitti_pose.py --test_seq [sequence_id] --dataset_dir /path/to/KITTI/odometry/set/ --output_dir /path/to/output/directory/ --ckpt_file /path/to/pre-trained/model/file/
```
A sample model trained on 5-frame snippets can be downloaded by
```bash
bash ./models/download_pose_model.sh
```
Then you can obtain predictions on, say `Seq. 9`, by running
```bash
python test_kitti_pose.py --test_seq 9 --dataset_dir /path/to/KITTI/odometry/set/ --output_dir /path/to/output/directory/ --ckpt_file models/model-100280
```

## Other implementations
[Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch) (by Clement Pinard)

## Disclaimer
This is the authors' implementation of the system described in the paper and not an official Google product.
