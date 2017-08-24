URL=http://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/models/kitti_depth_model.tar
OUTPUT_FILE=./models/kitti_depth_model.tar
wget -N $URL -O $OUTPUT_FILE
tar -xvf $OUTPUT_FILE -C ./models/
rm $OUTPUT_FILE