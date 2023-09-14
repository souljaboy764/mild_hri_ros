#!/usr/bin/python

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

rospy.init_node('webcam_view_node')
cap = cv2.VideoCapture("/dev/video8")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
if not cap.isOpened():
	rospy.signal_shutdown("Cannot open camera")

bridge = CvBridge()
rate = rospy.Rate(60)
image_pub = rospy.Publisher("webcam_image",Image,queue_size=10)
while not rospy.is_shutdown():
	# Capture frame-by-frame
	ret, frame = cap.read()
	# if frame is read correctly ret is True
	if not ret:
		rospy.signal_shutdown("Can't receive frame. Exiting ...")
	frame = np.transpose(frame, (1,0,2))[::-1]
	image_pub.publish(bridge.cv2_to_imgmsg(frame, encoding="bgr8"))
	rate.sleep()

# When everything done, release the capture
cap.release()
