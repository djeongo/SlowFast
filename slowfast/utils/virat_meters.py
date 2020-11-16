import os
import itertools

from slowfast.utils.meters import AVAMeter
from slowfast.utils.ava_eval_helper import (
    read_labelmap
)
import slowfast.utils.logging as logging
from sklearn.metrics import average_precision_score

import numpy as np
import cv2
import torch

logger = logging.get_logger(__name__)

class ViratMeter(AVAMeter):
    def __init__(self, overall_iters, cfg, mode):
        cfg.AVA.ANNOTATION_DIR = cfg.VIRAT.ANNOTATION_DIR
        cfg.AVA.GROUNDTRUTH_FILE = cfg.VIRAT.GROUNDTRUTH_FILE
        cfg.AVA.LABEL_MAP_FILE = cfg.VIRAT.LABEL_MAP_FILE
        cfg.AVA.FRAME_LIST_DIR = cfg.VIRAT.FRAME_LIST_DIR
        super().__init__(overall_iters, cfg, mode)

        self.all_yolo_outputs = []
        self.all_boxes = []
        self.all_inputs = []

    def update_stats(self, preds, ori_boxes, metadata, yolo_output, boxes, inputs, loss=None, lr=None):
        super().update_stats(preds, ori_boxes, metadata, loss, lr)

        self.all_yolo_outputs.append(yolo_output)
        self.all_boxes.append(boxes)
        self.all_inputs.append(inputs)

    def reset(self):
        """
        Reset the Meter.
        """
        super().reset()
        self.all_yolo_outputs = []
        self.all_boxes = []
        self.all_inputs = []

    def finalize_metrics(self, log=True):
        """
        Calculate and log the final AVA metrics.
        """
        super().finalize_metrics(log)

        all_yolo_outputs = torch.cat(self.all_yolo_outputs, dim=0)
        all_ori_boxes = torch.cat(self.all_ori_boxes, dim=0)
        all_metadata = torch.cat(self.all_metadata, dim=0)
        all_boxes = torch.cat(self.all_boxes, dim=0)
        all_inputs = list(itertools.chain.from_iterable(self.all_inputs))
        evaluate_yolo(
            all_yolo_outputs,
            all_ori_boxes.tolist(),
            all_metadata.tolist(),
            all_boxes.tolist(),
            all_inputs)

def get_iou(bb1, bb2):
    """
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def evaluate_yolo(all_yolo_outputs, all_ori_boxes, all_metadata, all_boxes, all_inputs):
    S = 7
    B = 2
    C = 12

    def determine_grid(x_center,y_center):
        x_grid = np.floor(x_center * S)
        y_grid = np.floor(y_center * S)
        grid_num = int(x_grid*S + y_grid)
        # logger.info(f"{x_center}, {y_center}, {grid_num}")
        return grid_num

    logger.info("all_yolo_outputs: {}".format(all_yolo_outputs.shape))
    yolo_view = all_yolo_outputs.view(all_yolo_outputs.shape[0], S*S, (B*5+C))

    # logger.info("all_ori_boxes: {}".format(all_ori_boxes.shape))
    # logger.info("all_ori_boxes: {}".format(torch.unique(all_ori_boxes[:,0])))
    # logger.info("all_metadata: {}".format(all_metadata.shape))

    video_index_in_order = [all_metadata[0]]

    boxes_in_order = [[]]
    for metadata, ori_boxes, boxes in zip(all_metadata, all_ori_boxes, all_boxes):
        if video_index_in_order[-1] == metadata:
            boxes_in_order[-1].append(boxes)
        else:
            video_index_in_order.append(metadata)
            boxes_in_order.append([boxes])
        # if tuple(metadata) not in metadatas:
        #     metadatas[tuple(metadata)] = []
        # metadatas[tuple(metadata)].append(metadata)

    logger.info("boxes_in_order: {}".format(len(boxes_in_order)))
    logger.info("video_index_in_order: {}".format(video_index_in_order))

    def draw_bbox(img, start_point, end_point, fname):
        color = (255,0,0)
        img = cv2.rectangle(
            one_img,
            start_point,
            end_point,
            color, 2)
        logger.info(f"Writing {fname}")
        cv2.imwrite(fname, img)

    for i, (index, box, img) in enumerate(zip(video_index_in_order, boxes_in_order, all_inputs)):
        if len(box) > 0:
            logger.info('img.shape: {}'.format(img.shape))
            one_img = np.transpose(img[:,0,:,:].cpu().numpy(), (1,2,0))
            # logger.info(one_img.shape)
            # logger.info(one_img)
            one_img = (255*one_img).astype(np.uint8)
            # logger.info(box)
            start_point = (int(box[0][1]), int(box[0][2]))
            end_point = (int(box[0][3]), int(box[0][4]))
            # draw_bbox(one_img, start_point, end_point, '/tmp/{}.jpg'.format(i))
        else:
            logger.info(f"Skipping {i}")
        # print(index, boxes)

    # Compute IOU and mAP
    W = 224
    H = 224
    maps = []
    for i in range(yolo_view.shape[0]):
        logger.debug('len(boxes_in_order): {}'.format(len(boxes_in_order)))
        logger.debug('i: {}'.format(i))
        logger.debug('boxes_in_order[i]: {}'.format(boxes_in_order[i]))
        box = boxes_in_order[i][0]
        x1_true, y1_true, x2_true, y2_true = box[1]/W, box[2]/W, box[3]/H, box[4]/H
        x_center, y_center = np.mean([x1_true, x2_true]), np.mean([y1_true, y2_true])
        grid = determine_grid(x_center, y_center)
        # Compute AP for each class
        average_precision_scores = []
        for c in range(C):
            y_true = np.zeros((S*S,))
            y_true[grid] = 1
            average_precision_scores.append(
                average_precision_score(y_true, yolo_view[i, :, B*5+c].cpu()))
        # mean AP (based on one image)
        maps.append(np.mean(average_precision_scores))

        # Compute IOU
        x_yolo, y_yolo, w_yolo, h_yolo = yolo_view[i, grid, 0:4].cpu().numpy()
        w_yolo = np.log(w_yolo)
        h_yolo = np.log(h_yolo)
        x1_yolo, y1_yolo, x2_yolo, y2_yolo = x_yolo-w_yolo/2, y_yolo-h_yolo/2, x_yolo+w_yolo/2, y_yolo+h_yolo/2
        yolo_box = {'x1':x1_yolo, 'y1':y1_yolo, 'x2':x2_yolo, 'y2':y2_yolo}
        true_box = {'x1':x1_true, 'y1':y1_true, 'x2':x2_true, 'y2':y2_true}
        if yolo_box['x1'] < yolo_box['x2'] and yolo_box['y1'] < yolo_box['y2'] and \
            true_box['x1'] < true_box['x2'] and true_box['y1'] < true_box['y2']:
            logger.debug('box: {}'.format(box))
            logger.debug('true_box: {}'.format(true_box))
            iou = get_iou(true_box, yolo_box)
            logger.info('iou: {}'.format(iou))

            img = all_inputs[i]
            one_img = np.transpose(img[:,0,:,:].cpu().numpy(), (1,2,0))
            one_img = (255*one_img).astype(np.uint8)
            start_point = (x1_yolo*W, y1_yolo*H)
            end_point = (x2_yolo*W, y2_yolo*H)
            # draw_bbox(one_img, start_point, end_point, '/tmp/{}-yolo.jpg'.format(i))

    logger.info("mAP: {}".format(np.mean(maps)))
