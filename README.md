# Live-Eye--Large-Scale-Object-Recognition
This project uses deep machine learning to perform object recognition at large scale.

We will be using state of the art deep machine learning tools and technologies to perform object recognition on more than 2,00,000 images.
The dataset used here is from imagenet and Microsoft COCO challenge 2015.

## Install Caffee

http://caffe.berkeleyvision.org/install_apt.html

    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler

    sudo apt-get install --no-install-recommends libboost-all-dev

## Install CUDA :

    sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb

    sudo apt-get update

    sudo apt-get install cuda

## Install BLAS - Atlas is default for caffe :

    sudo apt-get install libatlas-base-dev

## Install the python-dev package 
 To have the Python headers for building the pycaffe interface 

    sudo apt-get install python-dev

## Install cudnn
LINUX

    cd <installpath>
    export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH

    Add <installpath> to your build and link process by adding -I<installpath> to your compile
    line and -L<installpath> -lcudnn to your link line.

## Other dependencies.

    sudo cp include/cudnn.h /usr/local/cuda-7.5/include/

    sudo cp lib64/libcudnn* /usr/local/cuda-7.5/lib64/



## Before training.
   export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH


## Training command
    tools/train_net.py --gpu 0 --solver ./models/solvers/solver.prototxt --weights data/imagenet_models/VGG16.v2.caffemodel --imdb coco_2014_train --iters 500 --cfg ./experiments/cfgs/faster_rcnn_end2end.yml

## Testing command
    ./tools/test_net.py --gpu 0 --def ./models/coco/solvers/test.prototxt --net ./models/bigdata_coco_final.caffemodel --imdb coco_2015_test --cfg experiments/cfgs/faster_rcnn_end2end.yml 
