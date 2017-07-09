URL=http://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/pose_eval_data.tar
TAR_FILE=./kitti_eval/pose_eval_data.tar
wget -N $URL -O $TAR_FILE
tar -xvf $TAR_FILE -C ./kitti_eval/
rm $TAR_FILE