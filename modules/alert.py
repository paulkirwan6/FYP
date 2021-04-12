import socket
import time
from firebase import firebase
from modules.config import FIREBASE_PATH
from modules.config import FIREBASE_LOCAL

def update_figures_alert(violate_distance_count, violate_mask_count):
	print("Social Distancing Violations: {}\tFace Mask Violations: {}".format(violate_distance_count, violate_mask_count))
	fb = firebase.FirebaseApplication(FIREBASE_PATH)
	result = fb.patch(FIREBASE_LOCAL, {"facemask_violations":str(violate_mask_count),"distance_violations":str(violate_distance_count)})