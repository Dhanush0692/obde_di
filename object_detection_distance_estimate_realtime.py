# USAGE python3 real_time_object_detection.py

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from nimbusPython import NimbusClient

MAX_TEMP = 750
CRITCAL_TEMP = 800

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

prototxt = '/home/dhanush/Downloads/pi-object-detection-master/MobileNetSSD_deploy.prototxt.txt'
model = '/home/dhanush/Downloads/pi-object-detection-master/MobileNetSSD_deploy.caffemodel'
confidence_th = 0.2

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# start tof streaming
cli = NimbusClient.NimbusClient("192.168.0.69")


#MANUAL      = 0
#MANUAL_HDR  = 1
#AUTO        = 2
#AUTO_HDR    = 3
cli.setExposureMode(NimbusClient.AUTO_HDR)
cli.setExposure(10, framerate=10000)


# loop over the frames from the video stream
while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        header, (ampl, radial,x,y,z, conf) = cli.getImage(invalidAsNan=True)

        '''
        if temperature > MAX_TEMP:
                print("critical tempreature reached:", temperature/10)
        else:
                print("current temperature is", temperature/10)

        continue
        '''
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        #print("height and width of frame:", h,w)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence > confidence_th:
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        #print("box:", box)
                        (startX, startY, endX, endY) = box.astype("int")
                        d = (startX, startY, endX, endY)
                        #print(d)
                 
                        # centroid for bounding box
                        a = int((endX-startX)/2)
                        b = int((endY-startY)/2)
                        #print("centroid pixel co-ordinates:", a,b)
                        distance = radial[a][b]
                        #print("distance to centroid pixel:", distance)

                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(CLASSES[idx],
                                confidence * 100)+"  "+ "Distance: {} meters".format(str(round(distance,3)))

                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                     
                        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break

        # update the FPS counter
        fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
