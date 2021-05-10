# Path to host Firebase
FIREBASE_PATH = "https://fypdatabase-7f67a-default-rtdb.europe-west1.firebasedatabase.app/"

# Path to Child Node in Firebase
FIREBASE_LOCAL = "/Local/"

# base path to YOLO directory
YOLO_PATH = "yolov3"

# face mask detection models path
MODEL_PATH = "models"

# initialize minimum probability to filter weak person detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# initialise miminum probability to filter out weak face/mask detections
MASK_MIN_CONF = 0.3

# constant to get social distance in pixels based on person object height
# standing = 1.2 (assumes average height is approx. 1.7m )
# sitting = 2.5  (assumes average sitting height is approx. 0.8m )
HEIGHT_TO_DISTANCE_MULTIPLIER = 1.2