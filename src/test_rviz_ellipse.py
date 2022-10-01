import numpy as np
import pbdlib as pbd
import PyKDL
import rospy
import tf2_ros
from tf.transformations import *
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker, MarkerArray

from utils import *

import matplotlib
matplotlib.use("Qt5agg")

rospy.init_node("ellipse_test")
model = np.load('models/rocket_hri.npy', allow_pickle=True).item()
model.mu[:,:3], model.sigma[:,:3,:3],
viz_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
ellipses = MarkerArray()
for i in range(model.nb_states):
	marker = Marker()
	marker.ns = "HSMM"
	marker.id = i
	marker.lifetime = rospy.Duration(0.5)
	marker.frame_locked = False
	marker.action = Marker.ADD
	marker.header.frame_id = "base_footprint"
	marker.type = Marker.SPHERE
	
	marker.color.a = 1.
	marker.color.r = 1.
	marker.color.g = 1.

	(eigValues,eigVectors) = numpy.linalg.eig(model.sigma[i,:3,:3])
	
	eigx_n=-PyKDL.Vector(eigVectors[0,0],eigVectors[1,0],eigVectors[2,0])
	eigy_n=-PyKDL.Vector(eigVectors[0,1],eigVectors[1,1],eigVectors[2,1])
	eigz_n=-PyKDL.Vector(eigVectors[0,2],eigVectors[1,2],eigVectors[2,2])

	rot = PyKDL.Rotation (eigx_n,eigy_n,eigz_n)
	quat = rot.GetQuaternion()

	#painting the Gaussian Ellipsoid Marker
	marker.pose.orientation.x = quat[0]
	marker.pose.orientation.y = quat[1]
	marker.pose.orientation.z = quat[2]
	marker.pose.orientation.w = quat[3]
	
	marker.pose.position.x = model.mu[i,0]
	marker.pose.position.y = model.mu[i,1]
	marker.pose.position.z = model.mu[i,2]
	
	marker.scale.x = 20*eigValues[0]
	marker.scale.y = 20*eigValues[1]
	marker.scale.z = 20*eigValues[2]
	
	ellipses.markers.append(marker)
	
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')
plot_pbd(ax, model, alpha_hsmm=None)
plt.ion()
plt.show(block=False)
rate = rospy.Rate(100)
while not rospy.is_shutdown():
	for i in range(model.nb_states):
		ellipses.markers[i].header.stamp = rospy.Time.now()
	viz_pub.publish(ellipses)
	rate.sleep()
	plt.draw()
	plt.pause(0.001)
	if not plt.fignum_exists(fig.number):
		rospy.signal_shutdown('no figure')
		break

plt.close()