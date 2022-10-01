import cv2
import matplotlib.pyplot as plt
import numpy as np

import pbdlib as pbd
import qi

import sys
import time

from utils import *
from nuitrack_node import NuitrackWrapper

def prepare_axes(ax):
	ax.cla()
	# ax.view_init(15, 160)
	ax.set_xlim3d([-0.9, 0.1])
	ax.set_ylim3d([-0.1, 0.9])
	ax.set_zlim3d([-0.65, 0.35])
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

def plot_skeleton(ax, skeleton):
	for i in range(len(connections)):
		bone = connections[i]
		ax.plot(skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 0], skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 1], skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 2], 'r-', linewidth=5)
	ax.scatter(skeleton[:-1, 0], skeleton[:-1, 1], skeleton[:-1, 2], c='g', marker='o', s=100)
	ax.scatter(skeleton[-1:, 0], skeleton[-1:, 1], skeleton[-1:, 2], c='g', marker='o', s=200)

def plot_pbd(ax, model, mu_est_hsmm):
	# ax.scatter(mu_est_hsmm[-1:, 0], mu_est_hsmm[-1:, 1], mu_est_hsmm[-1:, 2], c='b', marker='o', s=200)
	pbd.plot_gmm3d(ax, model.mu[:,:3], model.sigma[:,:3,:3], color='blue', alpha=0.1)
	for i in range(model.nb_states):
		pbd.plot_gauss3d(ax, model.mu[i,:3], model.sigma[i,:3,:3],
					 n_points=20, n_rings=15, color='red', alpha=alpha_hsmm[i, -1])

######################
### Starting up Pepper
######################
session = qi.Session()
try:
	session.connect("tcp://192.168.100.122:9559")
except RuntimeError:
	print ("Can't connect to Naoqi.")
	sys.exit(1)

motion_service  = session.service("ALMotion")
if not motion_service.robotIsWakeUp():
	motion_service.wakeUp()
motion_service.setBreathEnabled('Body', False)

arm_joints = [1.57079633, -0.08726646, 1.57079633, 0.01745329] # default standing angle values
joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']
motion_service.setStiffnesses(joint_names, 1.0)
motion_service.setAngles(joint_names, arm_joints, 0.5)

model = np.load('models/handshake_hri_nojointvel.npy', allow_pickle=True).item()

nuitrack = NuitrackWrapper(horizontal=False)
nuitrack.update() # IDK Why but needed for the first time
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')
plt.ion()

count = 0
trajectory = []
input('')
while True:
	img, skeleton = nuitrack.update()
	if img is None:
		break
	cv2.imshow('Image', img)
	key = cv2.waitKey(1)
	if key == 27 or key == ord('q') or not plt.fignum_exists(fig.number):
		break
	
	if len(skeleton) == 0:
		continue
	
	# Giving ~10s ot the user to adjust pose for better visibility
	count += 1
	if count < 300:
		print(count)
		time.sleep(0.01)
		continue
	
	# Origin at right shoulder, like in the 
	skeleton -= skeleton[joints_idx["right_shoulder"]-1:joints_idx["right_shoulder"], :]
	skeleton = rotation_normalization(skeleton).dot(skeleton.T).T
	skeleton[:, 0] *= -1

	if len(trajectory) == 0:
		trajectory = [np.concatenate([skeleton[-2,:],np.zeros_like(skeleton[-2,:])])]
		continue
	if len(trajectory) > 80:
		break
	trajectory.append(np.concatenate([skeleton[-2,:], skeleton[-2,:] - trajectory[-1][:3]]))
	cond_traj = np.array(trajectory)
	alpha_hsmm = forward_variable(model, len(cond_traj), cond_traj, slice(0, 6))
	mu_est_hsmm, sigma_est_hsmm = model.condition(cond_traj, dim_in=slice(0, 6), dim_out=slice(6, 10))
	# motion_service.angleInterpolation(joint_names, mu_est_hsmm[-1].tolist(), 0.5, True)
	# motion_service.setAngles(joint_names, np.clip(mu_est_hsmm[-1], lower_bounds, upper_bounds).tolist(), 0.5)
	motion_service.setAngles(joint_names, mu_est_hsmm[-1].tolist(), 0.5)
	
	prepare_axes(ax)
	plot_skeleton(ax, skeleton)
	plot_pbd(ax, model, mu_est_hsmm)
	plt.pause(0.001)

plt.close()
motion_service.setAngles(joint_names, arm_joints, 0.1)
motion_service.setStiffnesses(joint_names, 0.0)
cv2.destroyAllWindows()
session.close()