#!/usr/bin/python

from PyNuitrack import py_nuitrack

import cv2
import numpy as np

from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from tf.transformations import *
import rospy
import ros_numpy

from utils import *

class NuitrackWrapper:
	def __init__(self, height=480, width=848, horizontal=True):
		self._height = height
		self._width = width
		self._horizontal = horizontal
		
		# Right Realsense
		# self.base2cam = euler_matrix(-1.5707, -0.0628318, -1.0053088)
		# self.base2cam[:3, 3] = np.array([-0.221, -0.651, 1.095])

		#Left Realsense
		self.base2cam = euler_matrix(-1.57072, -0., -2.13628)
		self.base2cam[:3, 3] = np.array([-0.13, 0.36, 1.15])

		self.init_nuitrack()
		print("Nuitrack Version:", self._nuitrack.get_version())
		print("Nuitrack License:", self._nuitrack.get_license())
	
	def init_nuitrack(self):
		self._nuitrack = py_nuitrack.Nuitrack()
		self._nuitrack.init()

		self._nuitrack.set_config_value("DepthProvider.Depth2ColorRegistration", "true")
		if not self._horizontal:
			self._nuitrack.set_config_value("DepthProvider.RotateAngle", "270")

		# Realsense Depth Module - force to 848x480 @ 30 FPS
		self._nuitrack.set_config_value("Realsense2Module.Depth.Preset", "5")
		self._nuitrack.set_config_value("Realsense2Module.Depth.RawWidth", str(self._width))
		self._nuitrack.set_config_value("Realsense2Module.Depth.RawHeight", str(self._height))
		self._nuitrack.set_config_value("Realsense2Module.Depth.ProcessWidth", str(self._width))
		self._nuitrack.set_config_value("Realsense2Module.Depth.ProcessHeight", str(self._height))
		self._nuitrack.set_config_value("Realsense2Module.Depth.FPS", "30")

		# Realsense RGB Module - force to 848x480 @ 30 FPS
		self._nuitrack.set_config_value("Realsense2Module.RGB.RawWidth", str(self._width))
		self._nuitrack.set_config_value("Realsense2Module.RGB.RawHeight", str(self._height))
		self._nuitrack.set_config_value("Realsense2Module.RGB.ProcessWidth", str(self._width))
		self._nuitrack.set_config_value("Realsense2Module.RGB.ProcessHeight", str(self._height))
		self._nuitrack.set_config_value("Realsense2Module.RGB.FPS", "30")


		devices = self._nuitrack.get_device_list()
		for i, dev in enumerate(devices):
			print(dev.get_name(), dev.get_serial_number())
			if i == 0:
				#dev.activate("ACTIVATION_KEY") #you can activate device using python api
				print(dev.get_activation())
				self._nuitrack.set_device(dev)

		self._nuitrack.create_modules()
		self._nuitrack.run()

	def reset_nuitrack(self):
		try:
			self._nuitrack.release()
		except:
			print("Could not release Nuitrack, just resetting it")

		self.init_nuitrack()

	def update(self):
		self._nuitrack.update()
		
		self._depth_img = self._nuitrack.get_depth_data()
		self._color_img = self._nuitrack.get_color_data()
		if not self._depth_img.size or not self._color_img.size:
			return None, []
		display_img = self._color_img.copy()

		data = self._nuitrack.get_skeleton()
		if len(data.skeletons)==0:
			return display_img, []

		skeleton = np.zeros((14,3))
		for bone in connections:
			j0 = data.skeletons[0][joints_idx[bone[0]]]
			j1 = data.skeletons[0][joints_idx[bone[1]]]
			x0 = (round(j0.projection[0]), round(j0.projection[1]))
			x1 = (round(j1.projection[0]), round(j1.projection[1]))
			cv2.line(display_img, x0, x1, line_color, 5)
		for i in range(1,15):
			x = (round(data.skeletons[0][i].projection[0]), round(data.skeletons[0][i].projection[1]))
			cv2.circle(display_img, x, 15, point_color, -1)
			skeleton[i-1] = data.skeletons[0][i].real * 0.001

		return display_img, skeleton

	def __del__(self):
		self._nuitrack.release()

class NuitrackROS(NuitrackWrapper):
	def __init__(self, height=480, width=848, camera_link='camera_color_optical_frame', horizontal=False):
		super().__init__(height=height, width=width, horizontal=horizontal)
		self._camera_link = camera_link
		intrinsics = intrinsics_horizontal if horizontal else intrinsics_vertical
		K = intrinsics[(self._width, self._height)]

		self._color_pub = rospy.Publisher('image_color', Image, queue_size=10)
		self._depth_pub = rospy.Publisher('image_depth', Image, queue_size=10)
		self._display_pub = rospy.Publisher('image_display', Image, queue_size=10)
		self._camerainfo_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=10)
		self._pc2_pub = rospy.Publisher('pointcloud2', PointCloud2, queue_size=10)
		self._viz_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

		self._camerainfo = CameraInfo()
		self._camerainfo.height = height
		self._camerainfo.width = width
		self._camerainfo.distortion_model = "plumb_bob"
		self._camerainfo.D = np.zeros(5)
		self._camerainfo.K = K.flatten()
		self._camerainfo.R = np.eye(3).flatten()
		self._camerainfo.P = np.eye(4)[:3, :]
		self._camerainfo.P[:3, :3] = K[:]
		self._camerainfo.P = self._camerainfo.P.flatten()


		# PointCloud stuff
		xx, yy = np.meshgrid(np.arange(height), np.arange(width))
		self._coords = 0.001 * np.matmul(np.concatenate([xx[:,:,None], yy[:,:,None], np.ones_like(xx[:,:,None])],axis=2), np.linalg.inv(K).T)
		self._pc2_msg = PointCloud2(
							height = 1,
							width = height*width,
							fields = [PointField(name=n, offset=i*np.dtype(np.float32).itemsize, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyzrgb')],
							point_step = np.dtype(np.float32).itemsize * 6,
							row_step = np.dtype(np.float32).itemsize * 6 * height * width
						)
		# self.pc2_data = np.zeros((height*width,6), dtype=[('x', np.float32),
		# 											('y', np.float32),
		# 											('z', np.float32),
		# 											('r', np.float32),
		# 											('g', np.float32),
		# 											('b', np.float32)
		# 											])

		self._markerarray_msg = MarkerArray()
		lines = []
		for i in range(14):
			marker = Marker()
			line_strip = Marker()
			line_strip.ns = marker.ns = "nuitrack_skeleton"
			marker.id = i
			line_strip.id = i + 14
			line_strip.lifetime = marker.lifetime = rospy.Duration(0.5)
			line_strip.frame_locked = marker.frame_locked = False
			line_strip.action = marker.action = Marker.ADD

			marker.type = Marker.SPHERE
			line_strip.type = Marker.LINE_STRIP

			line_strip.color.a = line_strip.color.r = marker.color.a = marker.color.g = 1
			line_strip.color.g = line_strip.color.b = marker.color.b = marker.color.r = 0

			marker.scale.x = marker.scale.y = marker.scale.z = 0.1
			line_strip.scale.x = 0.02

			line_strip.pose.orientation = marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)

			line_strip.points = [Point(), Point()]

			self._markerarray_msg.markers.append(marker)
			lines.append(line_strip)
		self._markerarray_msg.markers = self._markerarray_msg.markers + lines[:-1]

		self._header = Header(frame_id = camera_link, seq = 0)
		
		self._bridge = CvBridge()

	def update(self):
		display_img, skeleton = super().update()
		if display_img is None:
			return None, [], None
		self._header.seq += 1
		self._header.stamp = rospy.Time.now()

		def publish(publisher, image, encoding):
			if publisher.get_num_connections() > 0:
				msg = self._bridge.cv2_to_imgmsg(image, encoding=encoding)
				msg.header = self._header
				publisher.publish(msg)
		
		publish(self._color_pub, self._color_img, "bgr8")
		publish(self._display_pub, display_img, "bgr8")
		publish(self._depth_pub, self._depth_img, "passthrough")

		if self._camerainfo_pub.get_num_connections() > 0:
			self._camerainfo.header = self._header
			self._camerainfo_pub.publish(self._camerainfo)
		
		if self._pc2_pub.get_num_connections() > 0:						
			self._pc2_msg.data = np.concatenate([
										self._coords * self._depth_img[..., None].astype(np.float32),
										self._color_img[:, :, ::-1]/255.
									], axis=-1)\
									.reshape((-1,6)).astype(np.float32).tobytes()
			# self._pc2_msg.data = (self._coords * self._depth_img[..., None]).astype(np.float32).reshape((-1,3)).tobytes()
			self._pc2_msg.header = self._header
			self._pc2_pub.publish(self._pc2_msg)

		if self._viz_pub.get_num_connections() > 0 and len(skeleton)>0:
			for i in range(14):
				self._markerarray_msg.markers[i].pose.position.x = skeleton[i,0]
				self._markerarray_msg.markers[i].pose.position.y = -skeleton[i,1]
				self._markerarray_msg.markers[i].pose.position.z = skeleton[i,2]
				self._markerarray_msg.markers[i].header = self._header

			for i in range(len(connections)):
				bone = connections[i]
				self._markerarray_msg.markers[i+14].points[0].x = skeleton[joints_idx[bone[0]]-1,0]
				self._markerarray_msg.markers[i+14].points[0].y = -skeleton[joints_idx[bone[0]]-1,1]
				self._markerarray_msg.markers[i+14].points[0].z = skeleton[joints_idx[bone[0]]-1,2]
				self._markerarray_msg.markers[i+14].points[1].x = skeleton[joints_idx[bone[1]]-1,0]
				self._markerarray_msg.markers[i+14].points[1].y = -skeleton[joints_idx[bone[1]]-1,1]
				self._markerarray_msg.markers[i+14].points[1].z = skeleton[joints_idx[bone[1]]-1,2]
				self._markerarray_msg.markers[i+14].header = self._header
			self._viz_pub.publish(self._markerarray_msg)

		return display_img, skeleton, self._header.stamp

if __name__=="__main__":
	# nuitrack = NuitrackWrapper(horizontal=False)
	# nuitrack.update() # IDK Why but needed for the first time
	# fig = plt.figure()
	# ax = fig.add_subplot(projection='3d')
	# plt.ion()
	# count = 0
	# while True:
	# 	img, skeleton = nuitrack.update()
	# 	if img is None:
	# 		break
		
	# 	cv2.imshow('Image', img)
	# 	key = cv2.waitKey(1)
	# 	if key == 27 or key == ord('q'):
	# 		break
	# 	if key == 32:
	# 		nuitrack._mode = (nuitrack._mode + 1) % 2
		

	# 	ax.cla()
	# 	# ax.view_init(self.init_vertical, self.init_horizon)
	# 	ax.set_xlim3d([-0.9, 0.1])
	# 	ax.set_ylim3d([-0.1, 0.9])
	# 	ax.set_zlim3d([-0.65, 0.35])
	# 	ax.set_xlabel('X')
	# 	ax.set_ylabel('Y')
	# 	ax.set_zlabel('Z')
	# 	if len(skeleton) > 0:
	# 		skeleton -= skeleton[joints_idx["right_shoulder"]-1:joints_idx["right_shoulder"], :]
	# 		skeleton = rotation_normalization(skeleton).dot(skeleton.T).T
	# 		skeleton[:, 0] *= -1
	# 		# skeleton[:, 2] *= -1
	# 		for i in range(len(connections)):
	# 			bone = connections[i]
	# 			ax.plot(skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 0], skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 1], skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 2], 'r-', linewidth=5)
	# 		for i in range(14):
	# 			ax.scatter(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], c='g', marker='o', s=250)
	# 	plt.pause(1/30.)
	# 	if not plt.fignum_exists(fig.number):
	# 		break
	# plt.ioff()
	# plt.show()
	# plt.close()

	rospy.init_node("nuitrack_node")
	nuitrack = NuitrackROS(width=848, height=480, horizontal=False)
	rate = rospy.Rate(500)
	
	t = TransformStamped()
	t.header.frame_id = 'base_footprint'
	t.child_frame_id = 'hand'
	broadcaster = tf2_ros.StaticTransformBroadcaster()
	t.transform.rotation.w = 1.

	while not rospy.is_shutdown():
		display_img, skeleton, stamp = nuitrack.update()
		if len(skeleton)>0:
			hand_pose = skeleton[-1,:]
			hand_pose[1] *= -1
			hand_pose = nuitrack.base2cam[:3,:3].dot(hand_pose) + nuitrack.base2cam[:3,3]
			t.transform.translation.x = hand_pose[0]
			t.transform.translation.y = hand_pose[1]
			t.transform.translation.z = hand_pose[2]
			t.header.stamp = stamp
			broadcaster.sendTransform(t)
		rate.sleep()
		