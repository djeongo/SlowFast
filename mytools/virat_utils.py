import cv2
import json
import re

import glob
import os.path

LABELS = [
    "Person loading an Object to a Vehicle",
    "Person Unloading an Object from a Car/Vehicle",
    "Person Opening a Vehicle/Car Trunk",
    "Person Closing a Vehicle/Car Trunk",
    "Person getting into a Vehicle",
    "Person getting out of a Vehicle",
    "Person gesturing",
    "Person digging",
    "Person carrying an object",
    "Person running",
    "Person entering a facility",
    "Person exiting a facility"
]

def get_video_names():
    video_names = []
    for frame_path in glob.glob('/data/virat-v2/videos_original/*'):
        video_name = os.path.splitext(os.path.basename(frame_path))[0]
        video_names.append(video_name)
    return video_names

def parse_video_name(video_name):
    pattern1 = re.compile(r"VIRAT_S_(\d{2})(\d{2})(\d{2})")
    pattern2 = re.compile(r"VIRAT_S_(\d{2})(\d{2})(\d{2})_(\d{2})_(\d{6})_(\d{6})")
    match1 = pattern1.search(video_name)
    match2 = pattern2.search(video_name)

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


def get_video_info(video_name):
    # read files generated by get_resolution.sh
    video_info_path = '/data/virat-v2/video-info/{}.mp4.out'.format(video_name)
    video_info = json.load(open(video_info_path))
    return video_info

def get_frame(video_name, frame_number):
    video_frame = '{}_{:06d}'.format(video_name, frame_number)
    print(video_frame)
    return cv2.imread(f'/data/virat-v2/frames/{video_name}/{video_frame}.jpg')

def get_original_resolution(video):
    stream = get_video_info(video)['streams'][0]
    return stream['width'], stream['height']

def get_frame_resolution(video):
    frame = get_frame(video, 1)
    height, width, channels = frame.shape
    return width, height

def get_fps(video_name):
    r_frame_rate = get_video_info(video_name)['streams'][0]['r_frame_rate']
    a, b = r_frame_rate.split('/')
    return float(a)/float(b)