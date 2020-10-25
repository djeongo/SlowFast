import cv2
import sys
import time

annotation = sys.argv[1]
frames_path = "/home/pius/sdc1/data/VIRAT/VIRAT_S_000001/"

annotations = []
with open(annotation) as f:
    for line in f.readlines():
        annotations.append(line.split())

for i, annotation in enumerate(annotations):
    event_ID, event_type, duration, start_frame, end_frame, current_frame, lefttop_x, lefttop_y, width, height = annotation
    img_file = 'VIRAT_S_000001_{:06d}.jpg'.format(int(current_frame))
    img_path = frames_path+img_file
    print(f'{i}/{len(annotations)} {img_file}')
    image = cv2.imread(img_path)
    lefttop_x = int(lefttop_x)
    lefttop_y = int(lefttop_y)
    width = int(width)
    height = int(height)
    x_min = lefttop_x
    y_min = lefttop_y
    x_max = lefttop_x + width
    y_max = lefttop_y + height 
    new_image = cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (0,255,0), 2) 
    h, w, _ = new_image.shape
    new_h, new_w = h//6, w//6
    cv2.imwrite("/tmp/annotated-"+img_file, cv2.resize(new_image, (new_w, new_h)))
