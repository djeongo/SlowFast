import csv
import os.path
import glob
import pandas as pd
import re
from sklearn.model_selection import train_test_split

PATH="/home/pius/Downloads/VIRAT/annotations/*events.txt"
VIDEO_PATH="/home/pius/sdc1/data/VIRAT/"
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
            
def parse_video_name(video_name):
    pattern1 = re.compile(r"VIRAT_S_(\d{2})(\d{2})(\d{2})")
    pattern2 = re.compile(r"VIRAT_S_(\d{2})(\d{2})(\d{2})_(\d{2})_(\d{6})_(\d{6})")
    match1 = pattern1.search(video)
    match2 = pattern2.search(video)



    if match2:
        collection_group_id = match2.group(1)
        scene_id = match2.group(2)
        sequence_id = match2.group(3)
        segment_id = match2.group(4)
        starting_sec = match2.group(5)
        ending_sec = match2.group(6)
        
    if match1:
        collection_group_id = match1.group(1)
        scene_id = match1.group(2)
        sequence_id = match1.group(3)

    return int(collection_group_id), int(scene_id)

total_rows = 0
dfs = []
for path in glob.glob(PATH):
    video = os.path.basename(path.split('.')[0])
    collection_group_id, scene_id = parse_video_name(video)
    annotations = process_events_file(path)
    df = pd.DataFrame(annotations)
    df['video'] = video
    df['collection_group_id'] = collection_group_id
    df['scene_id'] = scene_id
    total_rows += len(df)
    dfs.append(df)

print(total_rows)
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

print("Total unique events: ", len(groups))

dfs_train, dfs_dev = train_test_split(dfs)
print("dfs_train:", len(dfs_train))
print("dfs_dev:", len(dfs_dev))

# Validate all frames exist
def validate_frames(dfs):
    for df in dfs_train:
        row = df.iloc[0]
        video_path = VIDEO_PATH+row['video']
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

validate_frames(dfs_train)
validate_frames(dfs_dev)

def format_for_slowfast(dfs):
    dfs_slowfast = []
    for video_id, df in enumerate(dfs):
        print('{}/{}'.format(video_id, len(dfs)))
        row = df.iloc[0]
        video_path = VIDEO_PATH + row['video']
        df['original_vido_id'] = df['video']
        df['video_id'] = video_id
        df['frame_id'] = df['current_frame']-df['start_frame']
        df['path'] = df.apply(lambda row: '{}/{}_{:06d}.jpg'.format(video_path, row['video'], row['current_frame']+1), axis=1)       
        df['labels'] = df.apply(lambda row: '"{}"'.format(row['event_type']-1), axis=1)
        dfs_slowfast.append(df[['original_vido_id', 'video_id', 'frame_id', 'path', 'labels']])
    return pd.concat(dfs_slowfast)


format_for_slowfast(dfs_train).to_csv('train.csv', sep=' ', index=False, quoting=csv.QUOTE_NONE)
format_for_slowfast(dfs_dev).to_csv('val.csv', sep=' ', index=False, quoting=csv.QUOTE_NONE)