ROOT_DIR=$1

echo "$ROOT_DIR"
ls "$ROOT_DIR"
cd "$ROOT_DIR" || exit 1

wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
wget http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
wget http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat


tar zxvf cars_test.tgz && rm cars_test.tgz
tar zxvf cars_train.tgz && rm cars_train.tgz
tar zxvf car_devkit.tgz && rm car_devkit.tgz
mv cars_test_annos_withlabels.mat devkit/