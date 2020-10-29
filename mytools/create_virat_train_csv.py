from virat_utils import parse_video_name
import csv
import glob
import os
import pandas as pd

def create_virat_train_csv(frames_dir, output_csv):
    dfs = []
    video_id = 0
    for video_id, video_path in enumerate(glob.glob(frames_dir+"/VIRAT*")):
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
    df.to_csv(output_csv, sep=' ', index=False, quoting=csv.QUOTE_NONE)

frames_dir_v1 = '/data/virat-v1/frames'
output_csv_v1 = '/data/virat-v1/train.csv' # CSV file containing "original_vido_id video_id frame_id path labels"
create_virat_train_csv(frames_dir_v1, output_csv_v1)
# frames_dir_v2 = '/data/virat-v2/'
# output_csv_v2 = '/data/virat-v2/train.csv' # CSV file containing "original_vido_id video_id frame_id path labels"
# create_virat_train_csv(frames_dir_v2, output_csv_v2)