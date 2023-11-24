from mild_hri.dataloaders.nuisi import *

from bp_hri.utils import *
import bp_hri

import torch
import numpy as np

import argparse

import os

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Point
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Pepper Teleop Tester')
# Results and Paths
parser.add_argument('--src', type=str, default='./data/buetepage/traj_data.npz', metavar='SRC',
					help='Path to read training and testing data (default: ./data/buetepage/traj_data.npz).')
args = parser.parse_args()

rospy.init_node('visualizer_node')

MODELS_FOLDER = os.path.join('models','bp_hri')
		
robot_vae_hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
robot_vae_config = robot_vae_hyperparams['vae_config'].item()
robot_vae = bp_hri.networks.VAE(**(robot_vae_config.__dict__)).to(device)
ckpt = torch.load(os.path.join(MODELS_FOLDER,'robot_vae.pth'))
robot_vae.load_state_dict(ckpt['model'])
robot_vae.eval()

human_tdm_hyperparams = np.load(os.path.join(MODELS_FOLDER,'tdm_hyperparams.npz'), allow_pickle=True)
human_tdm_config = human_tdm_hyperparams['tdm_config'].item()
human_tdm = bp_hri.networks.TDM(**(human_tdm_config.__dict__)).to(device)
ckpt = torch.load(os.path.join(MODELS_FOLDER,'tdm.pth'))
human_tdm.load_state_dict(ckpt['model_1'])
human_tdm.eval()

hri_hyperparams = np.load(os.path.join(MODELS_FOLDER,'hri_hyperparams.npz'), allow_pickle=True)
hri_config = hri_hyperparams['hri_config'].item()
hri = bp_hri.networks.HRIDynamics(**(hri_config.__dict__)).to(device)
ckpt = torch.load(os.path.join(MODELS_FOLDER,'hri.pth'))
hri.load_state_dict(ckpt['model'])
hri.eval()


dataset = PepperWindowDataset(args.src, train=True, window_length=robot_vae.window_size, downsample=1)

markerarray_msg = MarkerArray()
lines = []

num_joints = 3
for w in range(robot_vae.window_size):
	for i in range(num_joints):
		marker = Marker()
		line_strip = Marker()
		line_strip.ns = marker.ns = "nuitrack_skeleton"
		marker.header.frame_id = line_strip.header.frame_id = 'base_link'
		marker.id = i + w * num_joints
		line_strip.id = i + (robot_vae.window_size + w) * num_joints
		line_strip.lifetime = marker.lifetime = rospy.Duration(0.5)
		line_strip.frame_locked = marker.frame_locked = False
		line_strip.action = marker.action = Marker.ADD

		marker.type = Marker.SPHERE
		line_strip.type = Marker.LINE_STRIP

		line_strip.color.r = marker.color.g = 1
		line_strip.color.g = line_strip.color.b = marker.color.b = marker.color.r = 0
		line_strip.color.a = marker.color.a = 1/(robot_vae.window_size - w)

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
trajectory_msg.trajectory_start.joint_state.header.stamp = 'base_footprint'
trajectory_msg.trajectory_start.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
trajectory_msg.trajectory = []
traj = RobotTrajectory()
traj.joint_trajectory.header.frame_id = 'base_footprint'
traj.joint_trajectory.joint_names = trajectory_msg.trajectory_start.joint_state.name
traj.joint_trajectory.points = []
for i in range(robot_vae.window_size-1):
	point  = JointTrajectoryPoint()
	point.time_from_start = rospy.Duration(0.05)
	traj.joint_trajectory.points.append(point)
trajectory_msg.trajectory.append(traj)

trajectory_msg_gt = DisplayTrajectory()
trajectory_msg_gt.model_id = 'JulietteY20MP'
trajectory_msg_gt.trajectory_start.joint_state.header.stamp = 'base_footprint'
trajectory_msg_gt.trajectory_start.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
trajectory_msg_gt.trajectory = []
traj = RobotTrajectory()
traj.joint_trajectory.header.frame_id = 'base_footprint'
traj.joint_trajectory.joint_names = trajectory_msg.trajectory_start.joint_state.name
traj.joint_trajectory.points = []
for i in range(robot_vae.window_size-1):
	point  = JointTrajectoryPoint()
	point.time_from_start = rospy.Duration(0.05)
	traj.joint_trajectory.points.append(point)
trajectory_msg_gt.trajectory.append(traj)

trajectory_msg_ae = DisplayTrajectory()
trajectory_msg_ae.model_id = 'JulietteY20MP'
trajectory_msg_ae.trajectory_start.joint_state.header.stamp = 'base_footprint'
trajectory_msg_ae.trajectory_start.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
trajectory_msg_ae.trajectory = []
traj = RobotTrajectory()
traj.joint_trajectory.header.frame_id = 'base_footprint'
traj.joint_trajectory.joint_names = trajectory_msg.trajectory_start.joint_state.name
traj.joint_trajectory.points = []
for i in range(robot_vae.window_size-1):
	point  = JointTrajectoryPoint()
	point.time_from_start = rospy.Duration(0.05)
	traj.joint_trajectory.points.append(point)
trajectory_msg_ae.trajectory.append(traj)

rate = rospy.Rate(20)
robot_pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
robot_pub_ae = rospy.Publisher('move_group/display_autoencoded_path', DisplayTrajectory, queue_size=10)
robot_pub_gt = rospy.Publisher('move_group/display_groundtruth_path', DisplayTrajectory, queue_size=10)
human_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

actidx = np.hstack(dataset.actidx[2:3] - np.array([0,1]))
zr_hri = []
dh = []
for a in range(dataset.actidx[2][0], dataset.actidx[2][1]):
	x, label = dataset[a]
	label = torch.Tensor(label).to(device)
	x = torch.Tensor(x).to(device)
	x = torch.cat([x,label], dim=-1)
	print(label)
	seq_len = x.shape[0]
	x_p1_tdm = x[:,p1_tdm_idx]
	x_p1_vae = x[:,p1_vae_idx].reshape((seq_len, robot_vae.window_size, num_joints*6))[..., :num_joints*3].reshape((seq_len, robot_vae.window_size, num_joints, 3)).cpu().numpy()
	x_r2_hri = x[:,r2_hri_idx]
	x_r2_vae = x[:,r2_vae_idx]
	# with torch.no_grad():
	# 	_, _, d_x1_dist = human_tdm(x_p1_tdm, None)
	# 	hri_input = torch.concat([x_r2_hri, d_x1_dist.mean], dim=-1)
	# 	z_r2hri_dist, _ = hri(hri_input, None)
	# 	x_r2_gen = robot_vae._output(robot_vae._decoder(z_r2hri_dist.mean))
	# 	x_r2_gen = x_r2_gen.reshape((seq_len, robot_vae.window_size, robot_vae.num_joints))
		
	# 	x_r2_vaegen ,_,_ = robot_vae(x_r2_vae)

	# 	x_r2_gen = x_r2_gen.cpu().numpy()#.mean(0).cpu().numpy()
	# 	zr_hri.append(z_r2hri_dist.mean.cpu().numpy())
	# 	dh.append(d_x1_dist.mean.cpu().numpy())
	# 	x_r2_vaegen = x_r2_vaegen.reshape((seq_len, robot_vae.window_size, robot_vae.num_joints)).cpu().numpy()
	# 	x_r2_gt = x_r2_vae.reshape((seq_len, robot_vae.window_size, robot_vae.num_joints)).cpu().numpy()

	# np.savez_compressed('bp_latents_dataset.npz', zr_hri=np.array(zr_hri, dtype=object), dh=np.array(dh, dtype=object))
	current_robot_state = x_r2_hri[0:1]
	tdm_lstm_state = None
	hri_lstm_state = None
	for i in range(seq_len):
		with torch.no_grad():
			_, _, d_x1_dist, tdm_lstm_state = human_tdm(x_p1_tdm[i:i+1], tdm_lstm_state)
			hri_input = torch.concat([current_robot_state, d_x1_dist.mean], dim=-1)
			z_r2hri_dist, _, hri_lstm_state = hri(hri_input, hri_lstm_state)
			x_r2_gen = robot_vae._output(robot_vae._decoder(z_r2hri_dist.mean))
			x_r2_gen = x_r2_gen.reshape((1, robot_vae.window_size, robot_vae.num_joints))
			current_robot_state[0, :4] = x_r2_gen[0,1]
			# if i!=seq_len-1:
			# 	current_robot_state = x_r2_hri[i+1:i+2]
			
			x_r2_vaegen ,_,_ = robot_vae(x_r2_vae[i:i+1])

			x_r2_gen = x_r2_gen.cpu().numpy()#.mean(0).cpu().numpy()
			zr_hri.append(z_r2hri_dist.mean.cpu().numpy())
			dh.append(d_x1_dist.mean.cpu().numpy())
			x_r2_vaegen = x_r2_vaegen.reshape((1, robot_vae.window_size, robot_vae.num_joints)).cpu().numpy()
			x_r2_gt = x_r2_vae[i:i+1].reshape((1, robot_vae.window_size, robot_vae.num_joints)).cpu().numpy()
		print(i)
		stamp = rospy.Time.now()
		idx = i
		for w in range(robot_vae.window_size):
			for j in range(num_joints):
				markerarray_msg.markers[j + w * num_joints].pose.position.x = 0.7 - x_p1_vae[idx][w][j][0]
				markerarray_msg.markers[j + w * num_joints].pose.position.y = -0.1 - x_p1_vae[idx][w][j][1]
				markerarray_msg.markers[j + w * num_joints].pose.position.z = x_p1_vae[idx][w][j][2]
				if j!=0:
					line_idx = robot_vae.window_size*num_joints + j + w * (num_joints-1) -1
					markerarray_msg.markers[line_idx].points[0].x = 0.7 - x_p1_vae[idx][w][j-1][0]
					markerarray_msg.markers[line_idx].points[0].y = -0.1 - x_p1_vae[idx][w][j-1][1]
					markerarray_msg.markers[line_idx].points[0].z = x_p1_vae[idx][w][j-1][2]
					markerarray_msg.markers[line_idx].points[1].x = 0.7 - x_p1_vae[idx][w][j][0]
					markerarray_msg.markers[line_idx].points[1].y = -0.1 - x_p1_vae[idx][w][j][1]
					markerarray_msg.markers[line_idx].points[1].z = x_p1_vae[idx][w][j][2]

			if w == 0:
				trajectory_msg.trajectory_start.joint_state.position = x_r2_gen[0][w].tolist() + [1.57072, 0.0087, 0., 0.]
				trajectory_msg_gt.trajectory_start.joint_state.position = x_r2_gt[0][w].tolist() + [1.57072, 0.0087, 0., 0.]
				trajectory_msg_ae.trajectory_start.joint_state.position = x_r2_vaegen[0][w].tolist() + [1.57072, 0.0087, 0., 0.]
			else:
				trajectory_msg.trajectory[0].joint_trajectory.points[w-1].positions = x_r2_gen[0][w].tolist() + [1.57072, 0.0087, 0., 0.]
				trajectory_msg_gt.trajectory[0].joint_trajectory.points[w-1].positions = x_r2_gt[0][w].tolist() + [1.57072, 0.0087, 0., 0.]
				trajectory_msg_ae.trajectory[0].joint_trajectory.points[w-1].positions = x_r2_vaegen[0][w].tolist() + [1.57072, 0.0087, 0., 0.]
		
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
