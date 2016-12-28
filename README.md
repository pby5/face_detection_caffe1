#face_detection_caffe1

This is an implementation of "Face Detection with the Faster R-CNN" (arxiv 2016) in caffe.

We train with wider face dataset + vgg16 network pre-training + faster rcnn, more information you can refer to above paper.
Introduction

This repository is a fork from py-faster-rcnn. You can refer to py-faster-rcnn README.md and faster-rcnn README.md for more information.
Setup

1.put wider_face.py and fddb.py in lib/datasets/

2.put train.prototxt, test.prototxt, slover.prototxt in models/wider_face/VGG16/faster_rcnn_end2end/

3.put faster_rcnn_end2end.sh in experiments/scripts/

and other steps are same as py-faster-rcnn.

if you have questions about this code and you can issue to me.
