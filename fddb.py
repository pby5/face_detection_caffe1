# --------------------------------------------------------
# Fast R-CNN
import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
import PIL
from voc_eval import voc_eval
from fast_rcnn.config import cfg

class fddb(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'FDDB_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path)
        self._classes = ('__background__', # always index 0
                         'face')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # 32202 images and 393703 faces
        print 'fddb image size = {}'.format(len(self._image_index))
        # Default to roidb handler

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, self._image_set, index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'fold-all-out.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        image_index = []
        with open(image_set_file) as f:
            for line in f.readlines():
                x = line.split()
                path = x[0].strip()
                if path not in image_index:
                    image_index.append(path)
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'FDDB')
        
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_fddb_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries. for wider face dataset, 
        some bboxes are not positivate integer
        """
        # x1 -> [0, im_shape[1]]
        boxes[:, 0] = np.maximum(boxes[:, 0], 0)
        boxes[:, 0] = np.minimum(boxes[:, 0], im_shape[1])
        # y1 -> [0, im_shape[0]]
        boxes[:, 1] = np.maximum(boxes[:, 1], 0)
        boxes[:, 1] = np.minimum(boxes[:, 1], im_shape[0])
        # x2 -> [0, im_shape[1]]
        boxes[:, 2] = np.maximum(boxes[:, 2], 0)
        boxes[:, 2] = np.minimum(boxes[:, 2], im_shape[1])
        # y2 -> [0, im_shape[0]]
        boxes[:, 3] = np.maximum(boxes[:, 3], 0)
        boxes[:, 3] = np.minimum(boxes[:, 3], im_shape[0])
        
        return boxes
    
    def _load_fddb_annotation(self, index):
        """
        Load image and bounding boxes info from txt.
        """
        
        image_set_file = os.path.join(self._data_path, 'fold-all-out.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        bboxes = []
        with open(image_set_file) as f:
            for line in f.readlines():
                parsed_line = line.split()
                path = parsed_line[0].strip()
                if path == index:
                    x = float(parsed_line[1])
                    y = float(parsed_line[2])
                    w = float(parsed_line[3])
                    h = float(parsed_line[4])
                    bboxes.append([x,y,w,h])
                                 
        num_objs = len(bboxes)
                 
        boxes = np.zeros((num_objs, 4), dtype=np.int16)
        uboxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, bbox in enumerate(bboxes):
            x1 = float(bbox[0]) 
            y1 = float(bbox[1]) 
            x2 = float(bbox[0] + bbox[2]) 
            y2 = float(bbox[1] + bbox[3]) 
            cls = self._class_to_ind['face']
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        width  = PIL.Image.open(self.image_path_from_index(index)).size[0]
        height = PIL.Image.open(self.image_path_from_index(index)).size[1]
        im_shape = [height, width]   
        boxes = self._clip_boxes(boxes, im_shape) 
        print index        
        assert (boxes[:, 2] >= boxes[:, 0]).all(), boxes
        assert (boxes[:, 3] >= boxes[:, 1]).all(), boxes
        overlaps = scipy.sparse.csr_matrix(overlaps)
        uboxes = boxes
        return {'boxes' : uboxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}
                                                
 




