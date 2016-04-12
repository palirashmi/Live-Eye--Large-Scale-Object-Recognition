#!/usr/bin/env python

# Modified from original source. Original source credits listed below.

# --------------------------------------------------------
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
from os import listdir
from os.path import isfile, join

results = []
valDir = '/mnt/d/BigData/COCO/val2014'

CLASSES = ('__background__',
           'person', 'bicycle', 'car', 'motorcycle', 
           'airplane', 'bus','train', 'truck', 'boat', 
           'traffic light', 'fire hydrant', 'stop sign', 
           'parking meter', 'bench', 'bird', 'cat', 
           'dog', 'horse', 'sheep', 'cow', 
           'elephant', 'bear', 'zebra', 'giraffe', 
           'backpack', 'umbrella', 'handbag', 'tie', 
           'suitcase', 'frisbee', 'skis', 'snowboard', 
           'sports ball', 'kite', 'baseball bat', 'baseball glove', 
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 
           'wine glass', 'cup', 'fork', 'knife', 
           'spoon', 'bowl', 'banana', 'apple', 
           'sandwich', 'orange', 'broccoli', 'carrot', 
           'hot dog', 'pizza', 'donut', 'cake', 
           'chair', 'couch', 'potted plant', 'bed', 
           'dining table', 'toilet', 'tv', 'laptop', 
           'mouse', 'remote', 'keyboard', 'cell phone', 
           'microwave', 'oven', 'toaster', 'sink', 
           'refrigerator', 'book', 'clock', 'vase', 
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def vis_detections(im, class_name,cls_ind, dets,image_name,  thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    import json
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for i in inds:
        name = image_name
        name = name.replace('COCO_val2014_','')
        name = name.replace('.jpg','')
        item = {'image_id': int(name)}

        bbox = dets[i, :4]
        score = dets[i, -1]
        item['category_id'] = cls_ind
        item['bbox'] = [bbox[0],bbox[1],bbox[2],bbox[3]]
        item['score'] = score
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        #print item

        results.append(item)

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name, im_file):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls,cls_ind, dets,  image_name, thresh=CONF_THRESH,)


if __name__ == '__main__':
    results = []
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals


    prototxt = os.path.join(cfg.MODELS_DIR, 'coco',
                            'solvers', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              'bigdata_coco_final.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    cpu_mode = 0;

    if cpu_mode == 1:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    #print net.blobs['data']
    #print net.blobs['conv']
    for k, v in net.blobs.items():
        print k , ':' , v.data.shape 

    from time import time

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']

    fullSuite = True

    if fullSuite:
        im_names = [f for f in listdir(valDir) if isfile(join(valDir, f))]
        print len(im_names) , ' Will be processed '

    t = time()
    for im_name in im_names:
        #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        if not fullSuite:
            im_file = os.path.join(cfg.DATA_DIR, 'demo', im_name)
        else:
            im_file = os.path.join(valDir, im_name)
        demo(net, im_name, im_file)
    #print results
    print time() - t , ' Seconds in total'

    text_file = open("result.json", "w")
    text_file.write(str(results))
    text_file.close()
    
    if not fullSuite:
      plt.show()
    else:
      sys.exit(0)
