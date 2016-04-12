#!/usr/bin/env python


#%matplotlib inline
import matplotlib.pyplot as plt

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab


if __name__ == '__main__':


	pylab.rcParams['figure.figsize'] = (10.0, 8.0)

	annType = 'bbox'

	ground_truth = '/mnt/d/BigData/COCO/instances_train-val2014/annotations/instances_val2014.json' 
	generated_result = '/mnt/c/Users/Lavenger/git/py-faster-rcnn/tools/result.json'

	cocoGt = COCO(generated_result)

	cocoDt = cocoGt.loadRes(generated_result)

	cocoEval = COCOeval(cocoGt,cocoDt)
	cocoEval.params.imgIds  = imgIds
	cocoEval.params.useSegm = False
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()


