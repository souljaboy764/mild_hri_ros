import dataloaders
from vae import VAE

import torch
import numpy as np

import os

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Point
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rospy.init_node('visualizer_node')
window_size = 5
num_joints = 3
joint_dims = 3
downsample = 0.2
src = 'data/buetepage/traj_data.npz'
dataset = dataloaders.buetepage.PepperWindowDataset(src, train=False, window_length=window_size, downsample=downsample)
robot_pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
robot_pub_ae = rospy.Publisher('move_group/display_autoencoded_path', DisplayTrajectory, queue_size=10)
robot_pub_gt = rospy.Publisher('move_group/display_groundtruth_path', DisplayTrajectory, queue_size=10)
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

trajectory_msg_gt = DisplayTrajectory()
trajectory_msg_gt.model_id = 'JulietteY20MP'
trajectory_msg_gt.trajectory_start.joint_state.header.stamp = 'base_link'
trajectory_msg_gt.trajectory_start.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
trajectory_msg_gt.trajectory = []
traj = RobotTrajectory()
traj.joint_trajectory.header.frame_id = 'base_link'
traj.joint_trajectory.joint_names = trajectory_msg.trajectory_start.joint_state.name
traj.joint_trajectory.points = []
for i in range(window_size-1):
	point  = JointTrajectoryPoint()
	point.time_from_start = rospy.Duration(0.05)
	traj.joint_trajectory.points.append(point)
trajectory_msg_gt.trajectory.append(traj)

trajectory_msg_ae = DisplayTrajectory()
trajectory_msg_ae.model_id = 'JulietteY20MP'
trajectory_msg_ae.trajectory_start.joint_state.header.stamp = 'base_link'
trajectory_msg_ae.trajectory_start.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
trajectory_msg_ae.trajectory = []
traj = RobotTrajectory()
traj.joint_trajectory.header.frame_id = 'base_link'
traj.joint_trajectory.joint_names = trajectory_msg.trajectory_start.joint_state.name
traj.joint_trajectory.points = []
for i in range(window_size-1):
	point  = JointTrajectoryPoint()
	point.time_from_start = rospy.Duration(0.05)
	traj.joint_trajectory.points.append(point)
trajectory_msg_ae.trajectory.append(traj)


ckpt = torch.load(ckpt_path)
args_ckpt = ckpt['args']
z_dim = args_ckpt.latent_dim

model_h = VAE(**(ae_config.__dict__)).to(device)
model_h.load_state_dict(ckpt['model_h'])
model_h.eval()
model_r = VAE(**(robot_vae_config.__dict__)).to(device)
model_r.load_state_dict(ckpt['model_r'])
model_r.eval()
hsmm = ckpt['hsmm']

rate = rospy.Rate(20)
actidx = np.hstack(dataset.actidx - np.array([0,1]))
for a in actidx:
	x, label = dataset[a]
	seq_len = x.shape[0]
	dims_h = window_size*num_joints*joint_dims
	x_h = x[:, :dims_h].reshape((seq_len, window_size, num_joints, joint_dims))
	x_r = x[:, dims_h:].reshape((seq_len, window_size, 4))
	fwd_h = None
	with torch.no_grad():
		xr_rec, _, _ = model_r(torch.Tensor(x[:, dims_h:]).to(device))
		xr_rec = xr_rec.cpu().detach().numpy().reshape(-1, window_size, 4)

		if args_ckpt.cov_cond:
			zh_post = model_h(torch.Tensor(x[:, :dims_h]).to(device), dist_only=True)
			fwd_h = hsmm[label].forward_variable(demo=zh_post.mean, marginal=slice(0, z_dim))#, alpha_0=fwd_h), alpha_0=fwd_h)[:, -1]
			zr_cond = hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
												return_cov=False, data_Sigma_in=zh_post.covariance_matrix)
		else:
			zh_post = model_h(torch.Tensor(x[:, :dims_h]).to(device), encode_only=True)
			fwd_h = hsmm[label].forward_variable(demo=zh_post, marginal=slice(0, z_dim))#, alpha_0=fwd_h)
			zr_cond = hsmm[label].condition(zh_post, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
												return_cov=False, data_Sigma_in=None)
		xr_cond = model_r._output(model_r._decoder(zr_cond)).cpu().detach().numpy().reshape(-1, window_size, 4)
	for i in range(seq_len):
		stamp = rospy.Time.now()
		for w in range(window_size):
			for j in range(num_joints):
				markerarray_msg.markers[j + w * num_joints].pose.position.x = 0.7 - x_h[i][w][j][0]
				markerarray_msg.markers[j + w * num_joints].pose.position.y = -0.1 - x_h[i][w][j][1]
				markerarray_msg.markers[j + w * num_joints].pose.position.z = x_h[i][w][j][2]

				if j!=0:
					line_idx = window_size*num_joints + j + w * (num_joints-1) -1
					markerarray_msg.markers[line_idx].points[0].x = 0.7 - x_h[i][w][j-1][0]
					markerarray_msg.markers[line_idx].points[0].y = -0.1 - x_h[i][w][j-1][1]
					markerarray_msg.markers[line_idx].points[0].z = x_h[i][w][j-1][2]
					markerarray_msg.markers[line_idx].points[1].x = 0.7 - x_h[i][w][j][0]
					markerarray_msg.markers[line_idx].points[1].y = -0.1 - x_h[i][w][j][1]
					markerarray_msg.markers[line_idx].points[1].z = x_h[i][w][j][2]

			if w == 0:
				trajectory_msg.trajectory_start.joint_state.position = xr_cond[i][w].tolist() + [1.57072, 0.0087, 0., 0.]
				trajectory_msg_gt.trajectory_start.joint_state.position = x_r[i][w].tolist() + [1.57072, 0.0087, 0., 0.]
				trajectory_msg_ae.trajectory_start.joint_state.position = xr_rec[i][w].tolist() + [1.57072, 0.0087, 0., 0.]
			else:
				trajectory_msg.trajectory[0].joint_trajectory.points[w-1].positions = xr_cond[i][w].tolist() + [1.57072, 0.0087, 0., 0.]
				trajectory_msg_gt.trajectory[0].joint_trajectory.points[w-1].positions = x_r[i][w].tolist() + [1.57072, 0.0087, 0., 0.]
				trajectory_msg_ae.trajectory[0].joint_trajectory.points[w-1].positions = xr_rec[i][w].tolist() + [1.57072, 0.0087, 0., 0.]
		
		for j in range(len(markerarray_msg.markers)):
			markerarray_msg.markers[j].header.stamp = stamp

		trajectory_msg.trajectory[0].joint_trajectory.header.stamp = stamp
		trajectory_msg.trajectory_start.joint_state.header.stamp = stamp
		trajectory_msg_gt.trajectory[0].joint_trajectory.header.stamp = stamp
		trajectory_msg_gt.trajectory_start.joint_state.header.stamp = stamp
		trajectory_msg_ae.trajectory[0].joint_trajectory.header.stamp = stamp
		trajectory_msg_ae.trajectory_start.joint_state.header.stamp = stamp

		robot_pub.publish(trajectory_msg)
		robot_pub_gt.publish(trajectory_msg_gt)
		robot_pub_ae.publish(trajectory_msg_ae)
		human_pub.publish(markerarray_msg)
		rate.sleep()
		if rospy.is_shutdown():
			break
	if rospy.is_shutdown():
		break
