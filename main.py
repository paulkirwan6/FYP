# USAGE
# python social_distance_detector.py -i input/input_video.mp4
# python social_distance_detector.py -i pedestrians.mp4 -o output/output_video.avi

from tensorflow.keras.models import load_model
from modules import config
from modules.detection import detect_people
from modules.detection import detect_and_predict_mask
from modules.alert import update_figures_alert
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import time
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-hl", "--headless", type=str, default="false",
	help="path to disable displaying the screen")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.YOLO_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.YOLO_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.YOLO_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([config.MODEL_PATH, "deploy.prototxt"])
weightsPath = os.path.sep.join([config.MODEL_PATH, "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
mask_model_path = os.path.sep.join([config.MODEL_PATH, "mask_detector.model"])
maskNet = load_model(mask_model_path)

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
writer = None

# loop over the frames from the video stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# resize the frame and then detect people in it
	frame = imutils.resize(frame, width=700)
	results, avg_height = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))
	
	#Get minimum distance for violation
	min_distance = config.HEIGHT_TO_DISTANCE_MULTIPLIER*avg_height

	# initialize the set of indexes that violate the minimum social
	# distance
	violate_distance = set()

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
				if D[i, j] < min_distance:
					# update our violation set with the indexes of
					# the centroid pairs
					violate_distance.add(i)
					violate_distance.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if the index pair exists within the violation set, then
		# update the color
		if i in violate_distance:
			color = (0, 0, 255)

		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	violate_mask_count = 0

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		if label == "No Mask":
			violate_mask_count += 1

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)	

	# draw the total number of social distancing and face mask violations on the
	# output frame
	violate_distance_count = len(violate_distance)
	text = "Social Distancing Violations: {}    Face Mask Violations: {}".format(violate_distance_count, violate_mask_count)
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

	# Send alert to update the total number of violations
	update_figures_alert(violate_distance_count, violate_mask_count)

	# Check that headless mode has not been selected
	if args["headless"] == "false":
		# Display the frame
		cv2.imshow("Frame", frame)

	# press 'ESC' to quit
	if cv2.waitKey(30) & 0xff == 27:
		print("Esc pressed: shutting down program...")
		if writer is not None:
			writer.write(frame)
		break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)