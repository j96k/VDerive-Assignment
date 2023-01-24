# import the necessary packages
import numpy as np
import imutils
import time
import cv2
import os
from PIL import Image
import io
from io import StringIO


conf=0.5

thresh=0.3

yoloDir = "F:/Assigments/Accident/accident"
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([yoloDir, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
print(LABELS)
for i in LABELS:
    if i == "car":
        LABELS=i
print(LABELS)
print("Label is find:----------")  
             
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([yoloDir, "yolov3.weights"])
configPath = os.path.sep.join([yoloDir, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (1 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

input_video = "F:/Assigments/Accident/accident/carwinsplash.mp4"
vs = cv2.VideoCapture(input_video)
writer = None
(W, H) = (None, None)
total=2


while True:
        _, frame = vs.read()  # read the frame from camera
        frame = imutils.resize(frame, width=500)  # resize for display in ouput
        if W is None or H is None:
                (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > conf:
                                # scale the bounding box coordinates back relative to
                                # the size of the image, keeping in mind that YOLO
                                # actually returns the center (x, y)-coordinates of
                                # the bounding box followed by the boxes' width and
                                # height
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top
                                # and and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates,
                                # confidences, and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf,
                thresh)
        coordinates = []
        # ensure at least one detection exists
        if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw a bounding box rectangle and label on the frame
                        #color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,225,0), 2)
                        cord = (x,y,x+w,y+h)
                        coordinates.append(cord)
                        print(coordinates)

                        text = "{}: {:.4f}".format(LABELS,
                                confidences[i])
                        cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0), 2)
                    
                        
                        crop_img = frame[y:y+h, x:x+w]
                        cv2.imshow("cropped", crop_img)
                        cv2.waitKey(0)
                       

                        """for i in coordinates:
                            crop_img = frame[i[1]:i[3], i[0]:i[2]] #frame[y:y+h, x:x+w]
                            cv2.imshow("cropped", crop_img)"""

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break

cv2.release()
cv2.waitKey(0) #display the image untill any key press
# release the file pointers
print("[INFO] cleaning up...")
cv2.destroyAllWindows()        