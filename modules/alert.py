
FACEMASK_ALERT = "New breach in face mask regulations detected"
SOCIAL_DISTANCE_ALERT = "New breach in social distancing detected"

def new_violation_alert(violation):
	print("alert type: {} ".format(violation))
	
def update_figures_alert(violate_distance_count, violate_mask_count):
	print("Social Distancing Violations: {}\tFace Mask Violations: {}".format(violate_distance_count, violate_mask_count))