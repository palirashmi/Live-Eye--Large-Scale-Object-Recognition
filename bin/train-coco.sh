#Training command

tools/train_net.py --gpu 0 --solver ./models/solvers/solver.prototxt --weights data/imagenet_models/VGG16.v2.caffemodel --imdb coco_2014_train --iters 500 --cfg ./experiments/cfgs/faster_rcnn_end2end.yml

#Testing command

tools/test_net.py --gpu 0 --def ./models/solvers/test.prototxt --net ./models/bigdata_coco_final.caffemodel --imdb coco_2015_test --cfg experiments/cfgs/faster_rcnn_end2end.yml
