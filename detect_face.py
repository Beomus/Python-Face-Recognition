import numpy as np
import argparse
import cv2
import imutils
import time
from imutils.video import VideoStream

# setting up the argument parse and parse the arguments
ap = argparse.ArgumentParser()
"""
ap.add_argument('-i', '--image',
	required=True, help='Path to input image.')
"""
ap.add_argument('-p', '--prototxt',
	required=True, help="Path to Caffe 'deploy' prototxt file.")
ap.add_argument('-m', '--model',
	required=True, help='Path to Caffe pre-trained model')
ap.add_argument('-c', '--confidence', type=float, default=0.5,
	help='Minimum probability to filter weak detections.')
# Default to 0.5 but it is possible to change depending on your need

# Parsing the arguments
args = vars(ap.parse_args())

# Load the serialized model from directory
print("[INFO] Loading model...")
# Loading the model with the parameters
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
print('[INFO] Model loaded successfully.')


# Initialize video stream
print('[INFO] Starting video stream...')
video = VideoStream(src=0).start()
time.sleep(2.0)


# main loop to go over frames from the stream
run = True
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (600, 600))
while run:
	# Taking the frames from the video and resize it
	frame = video.read()
	frame = imutils.resize(frame, width=600)

	# take the frame dimensons and convert it into a blob
	# More info on "blob" (Binary Large OBject): https://answers.opencv.org/question/50025/what-exactly-is-a-blob-in-opencv/
	(height, width) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network to extract detections and prediction
	net.setInput(blob)
	detections = net.forward()

	# looping throught each detections
	for i in range(detections.shape[2]):
		# extract the confidence (or probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is greater than 
		# the minimum confidence
		if confidence < args['confidence']:
			continue

		# calculate the x, y coordinate for the box around the object
		box = detections[0, 0, i, 3:7] * np.array([width, height ,width ,height])
		(x_start, y_start, x_end, y_end) = box.astype("int")

		# draw the box accordingly
		# ensures that the starting position is not out of frame
		y = y_start - 10 if y_start - 10 > 10 else y_start + 10

		start_pos = (x_start, y_start)
		end_pos = (x_end, y_end)
		cv2.rectangle(frame, start_pos, end_pos, (0, 0, 255), 3)
		# params: main frame, start pos, end pos, color, thickness

		# the text in the box
		text = f"{round(confidence*100, 2)}%"
		cv2.putText(frame, text, (x_start, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

		out.write(frame)
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			run = False
print("[INFO] Exiting program...")
cv2.destroyAllWindows()
video.stop()
print("[INFO] Successfully exited")
