# base path to YOLO directory
YOLO_PATH = "yolov3"

# face mask detection models path
MODEL_PATH = "models"

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = False

# constant to get social distance in pixels based on person object height
HEIGHT_TO_DISTANCE_MULTIPLIER = 1.3