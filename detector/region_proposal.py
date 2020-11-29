# -*- coding: utf-8 -*-
import cv2
import numpy as np


class Regions:

    def __init__(self, image, boxes):
        self._image = image
        self._boxes = boxes

    def get_boxes(self):
        return self._boxes

    def get_patches(self, dst_size=None):
        patches = []
        for bb in self._boxes:
            (y1, y2, x1, x2) = bb
            patch = self._image[y1:y2, x1:x2]
            patch = cv2.resize(patch, (dst_size[1], dst_size[0]), interpolation=cv2.INTER_AREA)

            patches.append(patch)
        return np.asarray(patches)


class MSER:

    def detect(self, img):
        img_area = img.shape[0] * img.shape[1]
        mser = cv2.MSER_create(_delta=1, _min_area=int(img_area * 0.001))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        regions, _ = mser.detectRegions(img_gray)
        bounding_boxes = self._get_boxes(regions)
        regions = Regions(img, bounding_boxes)
        return regions

    def _get_boxes(self, regions):
        bbs = []
        for i, region in enumerate(regions):
            (x, y, w, h) = cv2.boundingRect(region.reshape(-1, 1, 2))
            bbs.append((y, y + h, x, x + w))

        return np.array(bbs)
