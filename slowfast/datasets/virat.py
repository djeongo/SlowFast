#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import numpy as np
import torch

from . import virat_helper as virat_helper
from . import cv2_transform as cv2_transform
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)

import cv2

@DATASET_REGISTRY.register()
class Virat(torch.utils.data.Dataset):
    """
    Virat Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        # self._use_bgr = cfg.VIRAT.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.VIRAT.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.VIRAT.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.VIRAT.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.VIRAT.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.VIRAT.TEST_FORCE_FLIP

        self._load_data(cfg)

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        (
            self._image_paths,
            self._video_idx_to_name,
        ) = virat_helper.load_image_lists(cfg, is_train=(self._split == "train"))

        # Loading annotations for boxes and labels.
        boxes_and_labels = virat_helper.load_boxes_and_labels(
            cfg, mode=self._split
        )

        logger.info('len(boxes_and_labels): {}'.format(len(boxes_and_labels)))
        logger.info('len(self._image_paths): {}'.format(len(self._image_paths)))
        assert len(boxes_and_labels) == len(self._image_paths)

        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = virat_helper.get_keyframe_data(boxes_and_labels, self._video_idx_to_name)

        # Calculate the number of used boxes.
        self._num_boxes_used = virat_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()

    def print_summary(self):
        logger.info("=== Virat dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        logger.debug("_images_and_boxes_preprocessing_cv2: imgs[0].shape: {}".format(imgs[0].shape))
        height, width, _ = imgs[0].shape # HWC

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for Virat, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            # imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
            #     imgs,
            #     min_size=self._jitter_min_scale,
            #     max_size=self._jitter_max_scale,
            #     boxes=boxes,
            # )
            imgs, boxes = cv2_transform.random_crop_list_include_boxes(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )

            logger.debug("after random_horizontal_flip: {}, len(iimgs):{}".format(imgs[0].shape, len(imgs)))
        elif self._split == "val":
            # Short side to test_scale. Non-local and STRG uses 256.
            # imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            # imgs = cv2_transform.crop_include_bbox(imgs, boxes, self._crop_size)
            # boxes = [
            #     cv2_transform.scale_boxes(
            #         self._crop_size, boxes[0], height, width
            #     )
            # ]
            # imgs, boxes = cv2_transform.spatial_shift_crop_list(
            #     self._crop_size, imgs, 1, boxes=boxes
            # )
            imgs, boxes = cv2_transform.random_crop_list_include_boxes(
                imgs, self._crop_size, order="HWC", boxes=boxes,
            )
            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            # imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            # imgs = cv2_transform.crop_include_bbox(imgs, boxes, self._crop_size)
            # boxes = [
            #     cv2_transform.scale_boxes(
            #         self._crop_size, boxes[0], height, width
            #     )
            # ]
            imgs, boxes = cv2_transform.random_crop_list_include_boxes(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )
            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]
        logger.debug("After convert image to CHW keeping BGR order: {}, len(imgs): {}".format(imgs[0].shape, len(imgs)))
        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        # imgs = [
        #     cv2_transform.color_normalization(
        #         img,
        #         np.array(self._data_mean, dtype=np.float32),
        #         np.array(self._data_std, dtype=np.float32),
        #     )
        #     for img in imgs
        # ]

        logger.debug("After Normalize images by mean and std.: {}, len(imgs): {}".format(imgs[0].shape, len(imgs)))

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        logger.debug("After Concat list of images to single ndarray.: {}, len(imgs): {}".format(imgs[0].shape, len(imgs)))

        # if not self._use_bgr:
        #     # Convert image format from BGR to RGB.
        #     imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        logger.debug("Returning: {}, len(imgs): {}".format(imgs[0].shape, len(imgs)))
        return imgs, boxes

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(
                imgs, self._crop_size, boxes=boxes
            )

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(
            boxes, self._crop_size, self._crop_size
        )

        return imgs, boxes

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        logger.debug("video_idx: {}".format(video_idx))
        logger.debug("sec_idx: {}".format(sec_idx))
        logger.debug("center_idx: {}".format(center_idx))
        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )
        logger.debug('seq: {}'.format(seq))

        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        labels = []
        for box_labels in clip_label_list:
            boxes.append(box_labels[0])
            labels.append(box_labels[1])
        boxes = np.array(boxes)
        # Score is not used.
        boxes = boxes[:, :4].copy()
        ori_boxes = boxes.copy()

        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.VIRAT.IMG_PROC_BACKEND
        )

        if self.cfg.VIRAT.IMG_PROC_BACKEND == "pytorch":
            # T H W C -> T C H W.
            imgs = imgs.permute(0, 3, 1, 2)
            # Preprocess images and boxes.
            imgs, boxes = self._images_and_boxes_preprocessing(
                imgs, boxes=boxes
            )
            # T C H W -> C T H W.
            imgs = imgs.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            # write_images(video_idx, sec, imgs, boxes)
            imgs, boxes = self._images_and_boxes_preprocessing_cv2(
                imgs, boxes=boxes
            )
            logger.debug('imgs.shape: {}'.format(imgs.shape))

        # Construct label arrays.
        label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
        for i, box_labels in enumerate(labels):
            # Virat label index starts from 1.
            for label in box_labels:
                # print(label)
                # if label == -1:
                #     continue
                # assert label >= 1 and label <= 80
                label_arrs[i][label-1] = 1
        logger.debug("Before utils.pack_pathway_output: {}".format(imgs.shape))
        imgs = utils.pack_pathway_output(self.cfg, imgs)
        metadata = [[video_idx, sec]] * len(boxes)
        logger.debug("After utils.pack_pathway_output: {}, len(imgs): {}".format(len(imgs), imgs[0].shape))

        extra_data = {
            "boxes": boxes,
            "ori_boxes": ori_boxes, # Collate function adds video_idx e.g., [video_idx, x, y, x, y]
            "metadata": metadata, # Need to enable DETECTION,
            "slowpath_imgs": imgs[0].cpu().clone()
        }

        return imgs, label_arrs, idx, extra_data

def write_images(video_idx, sec, imgs, boxes):
    logger.info('len(images): {}'.format(len(imgs)))
    logger.info('len(boxes): {}'.format(len(boxes)))

    img = imgs[len(imgs)-1]
    box = boxes[0]

    H, W, _ = img.shape
    logger.info('imgs[0]: {}'.format(img.shape))
    logger.info('boxes[0]: {}'.format(box))

    x1 = min([box[0] for box in boxes])
    y1 = min([box[1] for box in boxes])
    x2 = max([box[2] for box in boxes])
    y2 = max([box[3] for box in boxes])

    start_point = (int(x1*W), int(y1*H))
    end_point = (int(x2*W), int(y2*H))
    color = (255,0,0)
    img = cv2.rectangle(img, start_point, end_point, color, 2)
    cv2.imwrite('/tmp/orig-{}-{}.jpg'.format(video_idx, sec),  img)
