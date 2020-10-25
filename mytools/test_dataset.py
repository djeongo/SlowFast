from slowfast.datasets import Virat
from slowfast.datasets.build import build_dataset

from fvcore.common.config import CfgNode
cfg = CfgNode()
cfg.DATA = CfgNode()
cfg.DATA.PATH_TO_DATA_DIR = "/home/pius/sdc1/data/VIRAT"
cfg.DATA.PATH_PREFIX = "/home/pius/sdc1/data/VIRAT"
cfg.DATA.SAMPLING_RATE = 16
cfg.DATA.NUM_FRAMES = 4

cfg.DATA.MEAN = [0.45, 0.45, 0.45]
cfg.DATA.STD = [0.225, 0.225, 0.225]
cfg.DATA.TRAIN_JITTER_SCALES = [256, 320]
cfg.DATA.TRAIN_CROP_SIZE = 224
cfg.DATA.RANDOM_FLIP = True

cfg.MODEL = CfgNode()
cfg.MODEL.NUM_CLASSES = 35

cfg.VIRAT= CfgNode()
cfg.VIRAT.BGR = False
cfg.VIRAT.TRAIN_USE_COLOR_AUGMENTATION = False
cfg.VIRAT.TRAIN_PCA_JITTER_ONLY = False
cfg.VIRAT.TRAIN_PCA_EIGVAL = None
cfg.VIRAT.TRAIN_PCA_EIGVEC = None
cfg.VIRAT.TRAIN_LISTS = ['train.csv']

split = 'train'
dataset = build_dataset('Virat', cfg, split)

print(dataset)