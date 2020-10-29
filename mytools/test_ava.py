from slowfast.datasets import Ava
from slowfast.datasets.build import build_dataset

from fvcore.common.config import CfgNode
from slowfast.utils.parser import load_config, parse_args

class Args:
    def __init__(self):
        self.cfg_file = 'configs/AVA/SLOW_8x8_R50_SHORT_v2.yaml'
        self.opts = None
args = Args()

cfg = load_config(args)
cfg.AVA.FRAME_DIR = "/data/ava/frames/"

# Directory path for files of frame lists.
cfg.AVA.FRAME_LIST_DIR = (
    "/data/ava/frame_lists"
)

# Directory path for annotation files.
cfg.AVA.ANNOTATION_DIR = (
    "/data/ava/annotations/"
)

split = 'train'
dataset = build_dataset('AVA', cfg, split)
