from virat_utils import parse_video_name
import csv
import glob
import os
import pandas as pd

FRAMES_DIR = '/data/virat-v2/'
OUTPUT_CSV = '/data/virat-v2/train.csv' # CSV file containing "original_vido_id video_id frame_id path labels"

dfs = []
video_id = 0
for video_id, video_path in enumerate(glob.glob(FRAMES_DIR+"/VIRAT*")):
    video_name = os.path.basename(video_path)
    frames = []
    for frame_id, frame_jpg in enumerate(glob.glob(video_path+"/*.jpg")):
        frames.append({
            "original_vido_id":video_name,
            "video_id":video_id,
            "frame_id":frame_id,
            "path":frame_jpg,
            "labels":"\"\""})
    dfs.append(pd.DataFrame(frames))

df = pd.concat(dfs)
df.to_csv(OUTPUT_CSV, sep=' ', index=False, quoting=csv.QUOTE_NONE)