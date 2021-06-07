from firebase import firebase
from modules.config import FIREBASE_PATH
from modules.config import FIREBASE_LOCAL

# Send violation counts to Firebase via the specified path
# This data is stored in the JSON format
def update_figures_alert(violate_distance_count, violate_mask_count):
	fb = firebase.FirebaseApplication(FIREBASE_PATH)
	fb.patch(FIREBASE_LOCAL, {"facemask_violations":str(violate_mask_count),"distance_violations":str(violate_distance_count)})