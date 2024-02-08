import cv2
from utils import get_parking_spots_bboxes, empty_or_not, diff_img
import os
import numpy as np

# define the current path
current_path = os.getcwd()
print('current_path :', current_path)

# define our elements path
mask_path = 'data/mask_1920_1080.png'
video_path = 'data/parking_video.mp4'

# Read image, covert it to grayscale and threshold the mask
mask = cv2.imread(mask_path, -1)
gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Apply connected components methods to determine the spots in cropped image
connected_components = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
spots_status = [None for s in spots]
diffs = [None for d in spots]

previous_frame = None

# Read the video capture and get fps
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps*1)
frame_count = 0

while True:
    _, frame = cap.read()

    if frame_count % frame_interval == 0 and previous_frame is not None:
        for spot_idx, spot in enumerate(spots):
            x, y, w, h, cx, cy = spot
            crop_spot = frame[y: y+h, x: x+w]
            diffs[spot_idx] = diff_img(crop_spot, previous_frame[y: y+h, x: x+w])
        print(diffs)

    if frame_count % frame_interval == 0:
        if previous_frame is None:
            l = range(len(spots))
        else:
            l = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_idx in l:
            x, y, w, h, cx, cy = spots[spot_idx]

            crop_spot = frame[y: y+h, x: x+w]
            spot_status = empty_or_not(crop_spot)
            spots_status[spot_idx] = spot_status

    if frame_count % frame_interval == 0:
        previous_frame = frame.copy()

    for spot_idx, spot in enumerate(spots):
        spot_status = spots_status[spot_idx]
        x, y, w, h, cx, cy = spots[spot_idx]
        if spot_status:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.rectangle(frame, (130, 20), (630, 100), (0,0,0), -1)
    cv2.putText(frame, "Available spots : {} / {}".format(str(sum(spots_status)), str(len(spots_status))),
                (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)     # to fit the screen size
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
