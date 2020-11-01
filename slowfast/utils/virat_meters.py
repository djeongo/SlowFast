import os

from slowfast.utils.meters import AVAMeter
from slowfast.utils.ava_eval_helper import (
    read_labelmap
)

class ViratMeter(AVAMeter):
    def __init__(self, overall_iters, cfg, mode):
        cfg.AVA.ANNOTATION_DIR = cfg.VIRAT.ANNOTATION_DIR
        cfg.AVA.GROUNDTRUTH_FILE = cfg.VIRAT.GROUNDTRUTH_FILE
        cfg.AVA.LABEL_MAP_FILE = cfg.VIRAT.LABEL_MAP_FILE
        cfg.AVA.FRAME_LIST_DIR = cfg.VIRAT.FRAME_LIST_DIR
        super().__init__(overall_iters, cfg, mode)
