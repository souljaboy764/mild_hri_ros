from mild_hri.dataloaders. buetepage import *
from mild_hri.utils import *

import numpy as np

import argparse

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Point
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

parser = argparse.ArgumentParser(description='Pepper Teleop Tester')
# Results and Paths
parser.add_argument('--src', type=str, default='./data/buetepage/traj_data.npz', metavar='SRC',
					help='Path to read training and testing data (default: ./data/buetepage/traj_data.npz).')
args = parser.parse_args()

rospy.init_node('pepper_teleop_visualizer_node')
downsample = 0.2
window_size = 5
num_joints = 4
joint_dims = 3
pepper_dataset = PepperWindowDataset(args.src, train=True, window_length=window_size, downsample=downsample)
human_dataset = HHWindowDataset(args.src, train=True, window_length=window_size, downsample=downsample)

robot_pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
human_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
markerarray_msg = MarkerArray()
lines = []
for w in range(window_size):
	for i in range(num_joints):
		marker = Marker()
		line_strip = Marker()
		line_strip.ns = marker.ns = "nuitrack_skeleton"
		marker.header.frame_id = line_strip.header.frame_id = 'base_link'
		marker.id = i + w * num_joints
		line_strip.id = i + (window_size + w) * num_joints
		line_strip.lifetime = marker.lifetime = rospy.Duration(0.5)
		line_strip.frame_locked = marker.frame_locked = False
		line_strip.action = marker.action = Marker.ADD

		marker.type = Marker.SPHERE
		line_strip.type = Marker.LINE_STRIP

		line_strip.color.r = marker.color.g = 1
		line_strip.color.g = line_strip.color.b = marker.color.b = marker.color.r = 0
		line_strip.color.a = marker.color.a = 1/(window_size - w)

		marker.scale.x = marker.scale.y = marker.scale.z = 0.075
		line_strip.scale.x = 0.04

		line_strip.pose.orientation = marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)

		line_strip.points = [Point(), Point()]

		markerarray_msg.markers.append(marker)
		lines.append(line_strip)
	lines = lines[:-1]
markerarray_msg.markers = markerarray_msg.markers + lines

trajectory_msg = DisplayTrajectory()
trajectory_msg.model_id = 'JulietteY20MP'
trajectory_msg.trajectory_start.joint_state.header.stamp = 'base_link'
trajectory_msg.trajectory_start.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
trajectory_msg.trajectory = []
traj = RobotTrajectory()
traj.joint_trajectory.header.frame_id = 'base_link'
traj.joint_trajectory.joint_names = trajectory_msg.trajectory_start.joint_state.name
traj.joint_trajectory.points = []
for i in range(window_size-1):
	point  = JointTrajectoryPoint()
	point.time_from_start = rospy.Duration(0.05)
	traj.joint_trajectory.points.append(point)
trajectory_msg.trajectory.append(traj)

rate = rospy.Rate(20)
actidx = np.hstack(pepper_dataset.actidx - np.array([0,1]))
for a in actidx[::2]:
	q, _ = pepper_dataset[a]
	x, _ = human_dataset[a]
	seq_len = q.shape[0]
	dims_h = window_size*num_joints*joint_dims
	q = q[:, dims_h:].reshape((seq_len, window_size, 4))
	x = x[window_size:, dims_h:].reshape((-1, window_size, num_joints, joint_dims))
	
	for i in range(seq_len):
		stamp = rospy.Time.now()
		for w in range(window_size):
			for j in range(num_joints):
	
				markerarray_msg.markers[j + w * num_joints].pose.position.x = x[i][w][j][0]
				markerarray_msg.markers[j + w * num_joints].pose.position.y = x[i][w][j][1]
				markerarray_msg.markers[j + w * num_joints].pose.position.z = x[i][w][j][2] + 0.5

				if j!=0:
					line_idx = window_size*num_joints + j + w * (num_joints-1) -1
					markerarray_msg.markers[line_idx].points[0].x = x[i][w][j-1][0]
					markerarray_msg.markers[line_idx].points[0].y = x[i][w][j-1][1]
					markerarray_msg.markers[line_idx].points[0].z = x[i][w][j-1][2] + 0.5
					markerarray_msg.markers[line_idx].points[1].x = x[i][w][j][0]
					markerarray_msg.markers[line_idx].points[1].y = x[i][w][j][1]
					markerarray_msg.markers[line_idx].points[1].z = x[i][w][j][2] + 0.5

			if w == 0:
				trajectory_msg.trajectory_start.joint_state.position = q[i][w].tolist() + [1.57072, 0.0087, 0., 0.]
			else:
				trajectory_msg.trajectory[0].joint_trajectory.points[w-1].positions = q[i][w].tolist() + [1.57072, 0.0087, 0., 0.]

		for j in range(len(markerarray_msg.markers)):
			markerarray_msg.markers[j].header.stamp = stamp

		trajectory_msg.trajectory[0].joint_trajectory.header.stamp = stamp
		trajectory_msg.trajectory_start.joint_state.header.stamp = stamp
		robot_pub.publish(trajectory_msg)
		human_pub.publish(markerarray_msg)
		rate.sleep()
		if rospy.is_shutdown():
			break
	if rospy.is_shutdown():
		break
