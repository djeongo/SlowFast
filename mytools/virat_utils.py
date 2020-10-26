import re

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
