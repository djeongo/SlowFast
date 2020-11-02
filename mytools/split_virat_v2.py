import csv
import cv2
import os.path
import glob
import pandas as pd
import re
import virat_utils
from sklearn.model_selection import train_test_split

MAX_VIDEOS = -1
ANNOTATION_FILES="/data/virat-v2/raw-annotations/*events.txt" # file containing event id and bounding box
VIDEO_FRAMES_PATH="/data/virat-v2/frames/" # video frames path
FRAME_LISTS_PATH="/data/virat-v2/frame_lists/"
ANNOTATION_OUTPUT_PATH="/data/virat-v2/annotations/"

def process_events_file(events_file_path):
    cols = [
        'event_id',
        'event_type',
        'duration',
        'start_frame',
        'end_frame',
        'current_frame',
        'lefttop_x',
        'lefttop_y',
        'width',
        'height']
    annotations = []
    with open(events_file_path) as f:
        for line in f.readlines():
            annotation = line.split()
            annotations.append({x:int(y) for x,y in zip(cols, annotation)})
    return annotations

if False:
    total_rows = 0
    dfs = []
    for path in glob.glob(ANNOTATION_FILES):
        video = os.path.basename(path.split('.')[0])
        collection_group_id, scene_id = virat_utils.parse_video_name(video)
        annotations = process_events_file(path)
        df = pd.DataFrame(annotations)
        df.loc[:,'video'] = video
        df.loc[:,'collection_group_id'] = collection_group_id
        df.loc[:,'scene_id'] = scene_id
        total_rows += len(df)
        dfs.append(df)
        if MAX_VIDEOS > 0 and len(dfs) >= MAX_VIDEOS: break

    print('total_rows (of all annotation files): {}'.format(total_rows))
    df = pd.concat(dfs) #.set_index(['collection_group_id', 'scene_id', 'event_id'])
    groups = df.groupby(['video', 'collection_group_id', 'scene_id', 'event_id', 'start_frame', 'end_frame'])

    dfs = []
    for key, item in groups:
        df = groups.get_group(key)
        # Less than or equal to because the actual number of annotations could be less than
        # the duration of the event
        match = len(df) <= df.iloc[0]['duration']
        if not match:
            print(len(df), df.iloc[0]['duration'], df.iloc[0]['video'])
            print(df)
        dfs.append(df)
        if MAX_VIDEOS > 0 and len(dfs) >= MAX_VIDEOS: break

# Validate all frames exist
def validate_frames(dfs):
    for df in dfs_train:
        row = df.iloc[0]
        video_path = VIDEO_FRAMES_PATH+row['video']
        exists = os.path.exists(video_path)
        if not exists:
            print(video_path, exists)

        start_frame = row['start_frame']
        end_frame = row['end_frame']
        start_frame_path = '{}/{}_{:06d}.jpg'.format(video_path, row['video'], start_frame)
        exists = os.path.exists(start_frame_path)
        if not exists:
            print('start_frame:',start_frame_path, exists)

        end_frame_path = '{}/{}_{:06d}.jpg'.format(video_path, row['video'], end_frame)
        exists = os.path.exists(end_frame_path)
        if not exists:
            print('end_frame:',end_frame_path, exists)


# def format_for_slowfast(dfs):
#     dfs_slowfast = []
#     for video_id, df in enumerate(dfs):
#         # print('{}/{} formatting for slowfast'.format(video_id, len(dfs)))
#         row = df.iloc[0]
#         video_path = VIDEO_FRAMES_PATH + row['video']
#         df.loc[:,'original_vido_id'] = df['video']
#         df.loc[:,'video_id'] = video_id
#         df.loc[:,'frame_id'] = df['current_frame']-df['start_frame']
#         df.loc[:,'path'] = df.apply(lambda row: '{}/{}_{:06d}.jpg'.format(video_path, row['video'], row['current_frame']+1), axis=1)
#         df.loc[:,'labels'] = df.apply(lambda row: '"{}"'.format(row['event_type']-1), axis=1)
#         dfs_slowfast.append(df[['original_vido_id', 'video_id', 'frame_id', 'path', 'labels']])
#     return dfs_slowfast

VIDEO_IDS = {} # video->video_id

def get_frames(video):
    # video: Video name, e.g. VIRAT_S_040001_01_000448_001101
    # VIDEO_IDS[original_vido_id] = video_id
    frames = []
    for frame_id, frame in enumerate(glob.glob(f"{VIDEO_FRAMES_PATH}/{video}/*.jpg")):
        frames.append({
            'original_vido_id':video,
            'video_id':VIDEO_IDS[video],
            'frame_id':frame_id,
            'path':frame,
            'labels':"\"\""
        })
    return frames

def generate_frame_lists_files(videos):
    empty_dirs = 0
    dfs_frames = []
    for video in videos:
        frames = get_frames(video)
        df = pd.DataFrame(frames)
        if len(df) == 0:
            empty_dirs += 1
        dfs_frames.append(df)
    print('len(dfs_frames): {}'.format(len(dfs_frames)))
    if empty_dirs > 0:
        print("There are {} empty dirs!".format(empty_dirs))
    return dfs_frames

def generate_annotation_files_v2():
    dfs_annotations = []
    for video_id, event_file in enumerate(glob.glob(ANNOTATION_FILES)):
        video = os.path.basename(event_file.split('.')[0])
        if video in VIDEO_IDS:
            print("Annotations for the same video [{}] found!".format(video))
        VIDEO_IDS[video] = video_id
        annotations = process_events_file(event_file)
        df = pd.DataFrame(annotations)
        to_slowfast(video, df)

        dfs_annotations.append(df[['video_name', 'frame_sec', 'x1', 'y1', 'x2', 'y2', 'label', 'score']])

        if MAX_VIDEOS > 0 and video_id+1 >= MAX_VIDEOS: break

    print('len(dfs_annotations): {}'.format(len(dfs_annotations)))
    return dfs_annotations

def to_slowfast(video, df):
    orig_width, orig_height = virat_utils.get_original_resolution(video)
    cur_width, cur_height = virat_utils.get_frame_resolution(video)
    width_ratio = orig_width / cur_width
    height_ratio = orig_height / cur_height
    x1 = df['lefttop_x'] / width_ratio
    y1 = df['lefttop_y'] / height_ratio
    x2 = (df['lefttop_x'] + df['width']) / width_ratio
    y2 = (df['lefttop_y'] + df['height']) / height_ratio
    fps = virat_utils.get_fps(video)
    df.loc[:,'video_name'] = video
    df.loc[:,'frame_sec'] = (df['current_frame'] // fps).astype(int)
    df.loc[:,'x1'] = x1 / cur_width
    df.loc[:,'y1'] = y1 / cur_height
    df.loc[:,'x2'] = x2 / cur_width
    df.loc[:,'y2'] = y2 / cur_height
    df.loc[:,'label'] = df['event_type']
    df.loc[:,'score'] = 1

# def generate_annotation_files(dfs, dfs_slowfast):
#     # Process
#     #   *events*.csv files
#     # we want output to be:
#     #   video_id, frame_sec, x1, y1, x2, y2, label, score
#     #   * Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.
#     #   * score doesn't matter
#     # VIRAT bounding box format: 'lefttop_x', 'lefttop_y', 'width', 'height'
#     #
#     # Input
#     #   dfs: df per video
#     df_annotations = []
#     for df, df_slowfast in zip(dfs, dfs_slowfast):
#         width, height = get_resolution(df_slowfast.iloc[0]['path'])
#         df.loc[:,'video_name'] = df['video']
#         df.loc[:,'frame_sec'] = df['current_frame'] // FPS
#         df.loc[:,'x1'] = df['lefttop_x'] / width
#         df.loc[:,'y1'] = df['lefttop_y'] / height
#         df.loc[:,'x2'] = (df['lefttop_x'] + df['width']) / width
#         df.loc[:,'y2'] = (df['lefttop_y'] + df['height']) / height
#         df.loc[:,'label'] = df['event_type']-1
#         df.loc[:,'score'] = 1
#         df_annotations.append(df[['video_name', 'frame_sec', 'x1', 'y1', 'x2', 'y2', 'label', 'score']])
#     return df_annotations

# print("Total unique events: ", len(groups))

# dfs_train, dfs_dev = train_test_split(dfs)
# print("dfs_train:", len(dfs_train))
# print("dfs_dev:", len(dfs_dev))

# validate_frames(dfs_train)
# validate_frames(dfs_dev)

dfs_annotations = generate_annotation_files_v2() # Also generate VIDEO_IDS

dfs_annoations_train, dfs_annotations_dev = train_test_split(dfs_annotations)

dfs_annotations_map = {
    'train':{
        'dfs': dfs_annoations_train,
        'csv_annotations':'train_annotations.csv',
        'csv_frames':'train.csv',
    },
    'dev':{
        'dfs': dfs_annotations_dev,
        'csv_annotations':'dev_annotations.csv',
        'csv_frames':'dev.csv',
    }
}
for split in dfs_annotations_map.keys():
    dfs = dfs_annotations_map[split]['dfs']
    df = pd.concat(dfs)
    print(f"Spliit: {split}, len: {len(dfs)}")
    annotation_path = f'{ANNOTATION_OUTPUT_PATH}/{dfs_annotations_map[split]["csv_annotations"]}'
    print("Writing "+annotation_path)
    df.drop_duplicates().to_csv(annotation_path, sep=',', index=False, quoting=csv.QUOTE_NONE, header=False)

print('len(VIDEO_IDS): {}'.format(len(VIDEO_IDS)))
for split in dfs_annotations_map.keys():
    dfs = dfs_annotations_map[split]['dfs']
    dfs_frames = generate_frame_lists_files([df.iloc[0]['video_name'] for df in dfs])
    df_frames = pd.concat(dfs_frames) # TODO Split into train vs. dev based on annotation files
    train_frame_lists = f'{FRAME_LISTS_PATH}/{dfs_annotations_map[split]["csv_frames"]}'
    print('Writing to {}'.format(train_frame_lists))
    df_frames.to_csv(train_frame_lists, sep=' ', index=False, quoting=csv.QUOTE_NONE, header=False)

# dfs_slowfast_train = format_for_slowfast(dfs_train)
# dfs_slowfast_dev = format_for_slowfast(dfs_dev)

# df_annotations_train = generate_annotation_files(dfs_train, dfs_slowfast_train)
# df_annotations_dev = generate_annotation_files(dfs_dev, dfs_slowfast_dev)

# train_csv_path = f'{OUTPUT_PATH}/train.csv'
# val_csv_path = f'{OUTPUT_PATH}/val.csv'
# annotation_path = f'{ANNOTATION_OUTPUT_PATH}/train_annotation.csv'
# val_annotation_path = f'{ANNOTATION_OUTPUT_PATH}/val_annotation.csv'

# print("Writing "+train_csv_path)
# pd.concat(dfs_slowfast_train).to_csv(train_csv_path, sep=' ', index=False, quoting=csv.QUOTE_NONE)

# print("Writing "+val_csv_path)
# pd.concat(dfs_slowfast_dev).to_csv(val_csv_path, sep=' ', index=False, quoting=csv.QUOTE_NONE)

# print("Writing "+annotation_path)
# pd.concat(df_annotations_train).to_csv(annotation_path, sep=',', index=False, quoting=csv.QUOTE_NONE, header=False)

# print("Writing "+val_annotation_path)
# pd.concat(df_annotations_dev).to_csv(val_annotation_path, sep=',', index=False, quoting=csv.QUOTE_NONE, header=False)
