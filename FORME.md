# DATA
https://prior.allenai.org/projects/charades
## Charades
### Classes
77 - c077 Putting a pillow somewhere
79 - c079 Taking a pillow from somewhere
80 - c080 Throwing a pillow somewhere
76 - c076 Holding a pillow
75 - c075 Tidying up a blanket/s
### CSV Format
original_vido_id, video_id, frame_id, path, labels 

## ActEV: Activities in Extended Video
https://actev.nist.gov/

### Evaluation tasks
1) Activity Detection (AD)
    - given a target activity, a system automatically detects and temporally localizes all instances of the activity.
    - type of activity must be correct and the temporal overlap must fall within a minimal requirement.
2) Activity and Object Detection (AOD)
    - a system detects and temporally localizes all instances of the activity and spatially detects/localizes the people and/or objects associated with the target activity.
    - meet the temporal overlap criteria
    - in addition meet the spatial overlap of the identified objects during the activity instance.
3) Activity and Object Detection and Tracking (AODT).
    - a system detects and temporally localizes all instances of the activity, spatio-temporally detects/localizes the people and/or objects associated with the target activity, and properly assigns IDs the objects play in the activity. 

### Supported datasets
* Multiview Extended Video with Activities (MEVA) See the License and the README for context.
* VIRAT (The VIRAT Video Dataset)
    * There are multiple video clips from each scene, and each clip will contain zero or more instances of activities from 12 categories
    * VIRAT_S_XXYYZZ_KK_SSSSSS_TTTTTT.mp4
        * XX: Collection group ID
        * YY: Scene ID
        * ZZ: Sequence ID
        * KK: Segment ID
        * SSSSS: starting seconds
        * TTTTT: ending seconds
    * Download: https://viratdata.org/#getting-data
        * VIRAT Video Dataset consists of ground video dataset (~40G)
    * Annotations: https://gitlab.kitware.com/viratdata/viratannotations
        * Format
            Event File Columns 
            1: event ID        (unique identifier per event within a clip, same eid can exist on different clips)
            2: event type      (event type)
            3: duration        (event duration in frames)
            4: start frame     (start frame of the event)
            5: end frame       (end frame of the event)
            6: current frame   (current frame number)
            7: bbox lefttop x  (horizontal x coordinate of left top of bbox, origin is lefttop of the frame)
            8: bbox lefttop y  (vertical y coordinate of left top of bbox, origin is lefttop of the frame)
            9: bbox width      (horizontal width of the bbox)
            10: bbox height    (vertical height of the bbox)
        event ID    event type  duration    start_frame end_frame   current_frame   bbox
        0           1           59          3676        3734        3676            710 454 224 228 
    * Classes
        * 1 Person loading an Object to a Vehicle
        * 2 Person Unloading an Object from a Vehicle
        * 3 Person Opening a Vehicle Trunk 
        * 4 Person Closing a Vehicle Trunk
        * 5 Person getting into a Vehicle
        * 6 Person getting out of a Vehicle
        * 7 Person gesturing
        * 8 Person digging (Note: not existing in Release 2.0)
        * 9 Person Carrying an Object
        * 10 Person running
        * 11 Person entering a facility
        * 12 Person exiting a facility
    * Documentation:
        VIRAT_Video_Dataset_Release2.0_Introduction_v1.0.pdf
* Data access from mevadata.org: Accessing and using MEVA and MEVA Download Instructions
    * 250 hours of ground camera video, with additional resources such as UAV video, camera models, and a subset of 12.5 hours of annotated data.
* Data access from git and NIST : See actev-data-repo. Access credentials provided during signup

### Classes
https://actev.nist.gov/#tab_activities
37 activites
person_abandons_package	person_loads_vehicle	person_stands_up
person_closes_facility_door	person_transfers_object	person_talks_on_phone
person_closes_trunk	person_opens_facility_door	person_texts_on_phone
person_closes_vehicle_door	person_opens_trunk	person_steals_object
person_embraces_person	person_opens_vehicle_door	person_unloads_vehicle
person_enters_scene_through_structure	person_talks_to_person	vehicle_drops_off_person
person_enters_vehicle	person_picks_up_object	vehicle_picks_up_person
person_exits_scene_through_structure	person_purchases	vehicle_reverses
person_exits_vehicle	person_reads_document	vehicle_starts
hand_interacts_with_person	person_rides_bicycle	vehicle_stops
person_carries_heavy_object	person_puts_down_object	vehicle_turns_left
person_interacts_with_laptop	person_sits_down	vehicle_turns_right
vehicle_makes_u_turn

### Evaluation
* ActEV Scoring Software
  * https://github.com/usnistgov/ActEV_Scorer
* validate and score a system output file


# Install 
$ pip install av
$ pip install simplejson

$ git clone https://github.com/facebookresearch/detectron2 detectron2_repo
$ pip install -e detectron2_repo

in slowfast
$ python setup.py build develop

## VIRAT
# Train
```
python tools/run_net.py --cfg configs/VIRAT/SLOWFAST_16x8_R50.yaml DATA.PATH_TO_DATA_DIR ~/sdc1/data/VIRAT DATA.PATH_PREFIX / NUM_GPUS 1 TRAIN.BATCH_SIZE 1 TEST.ENABLE False
```
# Test
```
python tools/run_net.py --cfg configs/VIRAT/SLOWFAST_16x8_R50.yaml DATA.PATH_TO_DATA_DIR ~/sdc1/data/VIRAT DATA.PATH_PREFIX / NUM_GPUS 1 TRAIN.ENABLE False TEST.ENABLE True
```

# Train

## Charades
### Non-multigrid
```
python tools/run_net.py --cfg configs/Charades/SLOWFAST_16x8_R50.yaml DATA.PATH_TO_DATA_DIR /home/ubuntu/data/charades DATA.PATH_PREFIX /home/ubuntu/data/charades/Charades_v1_rgb NUM_GPUS 1 TRAIN.BATCH_SIZE 4
```
### Multigrid
* Seems to require more than 1 GPU
    python tools/run_net.py --cfg configs/Charades/SLOWFAST_16x8_R50_multigrid.yaml DATA.PATH_TO_DATA_DIR /home/ubuntu/data/charades DATA.PATH_PREFIX /home/ubuntu/data/charades/Charades_v1_rgb NUM_GPUS 1 TRAIN.BATCH_SIZE 16 

# Test
python tools/run_net.py --cfg configs/Charades/SLOWFAST_16x8_R50_multigrid.yaml DATA.PATH_TO_DATA_DIR /home/ubuntu/data/charades DATA.PATH_PREFIX /home/ubuntu/data/charades NUM_GPUS 1 TEST.CHECKPOINT_FILE_PATH SLOWFAST_16x8_R50_multigrid.pkl TRAIN.ENABLE False


https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md
