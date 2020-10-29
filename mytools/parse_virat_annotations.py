# Parse VIRAT annotations
# and generate _C.VIRAT.TRAIN_GT_BOX_LISTS = ["virat-gt-box-lists.csv"]

# https://actev.nist.gov/trecvid20#tab_activities
# VIRAT Activity Name (Original)	VIRAT Activity Name 2020
# Closing	
# person_closes_facility_or_vehicle_door
# Closing_Trunk	person_closes_trunk
# DropOff_Person_Vehicle	vehicle_drops_off_person
# Entering	person_enters_facility_or_vehicle
# Exiting	person_exits_facility_or_vehicle
# Interacts	person_interacts_object
# Loading	person_loads_vehicle
# Open_Trunk	person_opens_trunk
# Opening	
# person_opens_facility_or_vehicle_door
# Person_Person_Interaction	person_person_interaction
# PickUp	person_pickups_object
# PickUp_Person_Vehicle	vehicle_picks_up_person
# Pull	person_pulls_object
# Push	person_pushs_object
# Riding	person_rides_bicycle
# SetDown	person_sets_down_object
# Talking	person_talks_to_person
# Transport_HeavyCarry	person_carries_heavy_object
# Unloading	person_unloads_vehicle
# activity_carrying	person_carries_object
# activity_crouching	person_crouches
# activity_gesturing	person_gestures
# activity_running	person_runs
# activity_sitting	person_sits
# activity_standing	person_stands
# activity_walking	person_walks
# specialized_talking_phone	person_talks_on_phone
# specialized_texting_phone	person_texts_on_phone
# specialized_using_tool	person_uses_tool
# vehicle_moving	vehicle_moves
# vehicle_starting	vehicle_starts
# vehicle_stopping	vehicle_stops
# vehicle_turning_left	vehicle_turns_left
# vehicle_turning_right	vehicle_turns_right
# vehicle_u_turn	vehicle_makes_u_turn
import glob
import yaml

FPS = 30
VIRAT_ACTIVITY_NAMES = [
    'Closing',
    'Closing_Trunk',
    'DropOff_Person_Vehicle',
    'Entering',
    'Exiting',
    'Interacts',
    'Loading',
    'Open_Trunk',
    'Opening',
    'Person_Person_Interaction',
    'PickUp',
    'PickUp_Person_Vehicle',
    'Pull',
    'Push',
    'Riding',
    'SetDown',
    'Talking',
    'Transport_HeavyCarry',
    'Unloading',
    'activity_carrying',
    'activity_crouching',
    'activity_gesturing',
    'activity_running',
    'activity_sitting',
    'activity_standing',
    'activity_walking',
    'specialized_talking_phone',
    'specialized_texting_phone',
    'specialized_using_tool',
    'vehicle_moving',
    'vehicle_starting',
    'vehicle_stopping',
    'vehicle_turning_left',
    'vehicle_turning_right',
    'vehicle_u_turn'
]
ANNOTATION_PATH='/home/ubuntu/viratannotations/train'



def parse_annotation(annotation):
    for item in annotation:
        if 'act' in item:
            # print(item['act'].keys())
            act2 = item['act']['act2']
            activity = list(act2.keys())[0]
            if activity not in VIRAT_ACTIVITY_NAMES:
                print('Found', activity)
            timespan = item['act']['timespan'][0]
            start_time = timespan['tsr0'][0]
            end_time = timespan['tsr0'][1]
            start_frame = FPS*start_time
            end_frame = FPS*end_time
            print(activity, start_frame, end_frame)
            # print('act2', item['act']['act2'])
            # print('id2', item['act']['id2'])
            # print('timespan', item['act']['timespan'])
            # print('src', item['act']['src'])
            # print('actors', item['act']['actors'])

for annotation in glob.glob(ANNOTATION_PATH+"/*activities*"):
    if "VIRAT_S_000000" not in annotation:
        continue
    print(annotation)
    y = yaml.load(open(annotation))
    parse_annotation(y)
