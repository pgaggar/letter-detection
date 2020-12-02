import cv2
import numpy as np


def non_max_suppression(boxes, patches, probs, overlap_threshold=0.3):
    """
    Parameters:
        boxes (ndarray of shape (N, 4))
        patches (ndarray of shape (N, 32, 32, 1))
        probs (ndarray of shape (N,))
        overlap_threshold (float)

    Reference: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """
    if len(boxes) == 0:
        return [], [], []

    boxes = np.array(boxes, dtype="float")

    pick = []
    y1 = boxes[:, 0]
    y2 = boxes[:, 1]
    x1 = boxes[:, 2]
    x2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of
        # picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the
        # smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater than the
        # provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int"), patches[pick], probs[pick]


def convert_num_to_char(num):
    if num <= 10:
        return str(num - 1)
    elif num <= 36:
        num = num - 11
        return chr(ord('A') + num)
    else:
        num = num - 37
        return chr(ord('a') + num)


def _get_thresholded_boxes(bbs, patches, probs, threshold):
    """
    Parameters:
        regions (Regions)
    """
    bbs = bbs[probs > threshold]
    patches = patches[probs > threshold]
    probs = probs[probs > threshold]
    return bbs, patches, probs


class Detector:

    def __init__(self, model, region_proposer):
        """
        Parameters:
            model_file (str)
            region_proposer (MserRegionProposer)
        """
        self._model = model
        self._region_proposer = region_proposer

    def run(self, image, threshold=0.7, nms_threshold=0.3):
        """Public function to run the DigitSpotter.

        Parameters
        ----------
        image : str
            filename of the test image
            
        Returns
        ----------
        bbs : ndarray, shape of (N, 4)
            detected bounding box. (y1, y2, x1, x2) ordered.
        
        probs : ndarray, shape of (N,)
            evaluated score for the DigitSpotter and test images on average precision. 
    
        Examples
        --------
        """

        # 1. Get candidate patches
        candidate_regions = self._region_proposer.detect(image)
        patches = candidate_regions.get_patches(dst_size=self._model.input_shape)
        boxes = candidate_regions.get_boxes()

        # 2. Run pre-trained classifier
        proba = self._model.predict(patches)
        probs = np.sum(proba[:, 1:63], axis=1)
        # # # 4. Thresholding
        bbs, patches, probs = _get_thresholded_boxes(boxes, patches, probs, threshold)

        # 3. non-maxima-suppression
        bbs, patches, probs = non_max_suppression(bbs, patches, probs, nms_threshold)

        y_pred = probs_ = []
        if len(patches) > 0:
            modified_patches = []
            kernel = np.ones((2, 1), np.uint8)
            for patch in patches:
                mod = cv2.erode(patch, kernel, iterations=1)
                mod = cv2.dilate(mod, kernel, iterations=1)
                modified_patches.append(mod)
            probs_ = self._model.predict(patches)
            y_pred = np.argmax(probs_, axis=1)

        # 4. Classify text and annotate image
        print("=====Detected Text======")
        for i, bb in enumerate(bbs):
            if 1 <= y_pred[i] <= 62 and max(probs_[i]) >= 0.3:
                y1, y2, x1, x2 = bb
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                pred = y_pred[i]
                msg = convert_num_to_char(pred)
                cv2.putText(image, msg, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                print(msg)
        # cv2.imshow("Final Output", image)
        # cv2.waitKey(0)

        return image


if __name__ == "__main__":
    pass
