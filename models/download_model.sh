URL=http://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/models/kitti_cs_model.tar.gz
OUTPUT_FILE=./models/kitti_cs_model.tar.gz
wget -N $URL -O $OUTPUT_FILE
tar -xvf $OUTPUT_FILE -C ./models/
rm $OUTPUT_FILE