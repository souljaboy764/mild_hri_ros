import cv2
import matplotlib.pyplot as plt
import numpy as np
import pbdlib as pbd
from qibullet import SimulationManager
import time
from utils import *
from nuitrack_node import NuitrackROS, NuitrackWrapper

simulation_manager = SimulationManager()
client_id = simulation_manager.launchSimulation(gui=True)
pepper = simulation_manager.spawnPepper(
	client_id,
	translation=[0, 0, 0],
	quaternion=[0, 0, 0, 1],
	spawn_ground_plane=True)
joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
arm_joints = [ 1.57079633, 0.08726646, -1.57079633, -0.01745329, 0., 1.57079633, -0.08726646, 1.57079633, 0.01745329, 0.]
pepper.setAngles(joint_names, arm_joints, 1.0)
joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']

model = np.load('models/handshake_hri.npy', allow_pickle=True).item()
nuitrack = NuitrackROS(horizontal=False)
nuitrack.update() # IDK Why but needed for the first time
fig = plt.figure(figsize=(10,10))
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
	if key == 32:
		nuitrack._mode = (nuitrack._mode + 1) % 2
	count += 1
	if count < 15:
		time.sleep(1)
		continue
	
	if len(skeleton) == 0:
		continue
	skeleton -= skeleton[joints_idx["right_shoulder"]-1:joints_idx["right_shoulder"], :]
	skeleton = rotation_normalization(skeleton).dot(skeleton.T).T
	skeleton[:, 0] *= -1

	if len(trajectory) == 0:
		trajectory = [np.concatenate([skeleton[-1,:],np.zeros_like(skeleton[-1,:])])]
		continue

	trajectory.append(np.concatenate([skeleton[-1,:], skeleton[-1,:] - trajectory[-1][:3]]))
	cond_traj = np.array(trajectory)
	alpha_hsmm = forward_variable(model, len(cond_traj), cond_traj, slice(0, 6))
	mu_est_hsmm, sigma_est_hsmm = model.condition(cond_traj, dim_in=slice(0, 6), dim_out=slice(6, 10))
	
	ax.cla()
	# ax.view_init(15, 160)
	ax.set_xlim3d([-0.9, 0.1])
	ax.set_ylim3d([-0.1, 0.9])
	ax.set_zlim3d([-0.65, 0.35])
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	for i in range(len(connections)):
		bone = connections[i]
		ax.plot(skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 0], skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 1], skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 2], 'r-', linewidth=5)
	ax.scatter(skeleton[:-1, 0], skeleton[:-1, 1], skeleton[:-1, 2], c='g', marker='o', s=100)
	ax.scatter(skeleton[-1:, 0], skeleton[-1:, 1], skeleton[-1:, 2], c='g', marker='o', s=200)
	# ax.scatter(mu_est_hsmm[-1:, 0], mu_est_hsmm[-1:, 1], mu_est_hsmm[-1:, 2], c='b', marker='o', s=200)
	pepper.setAngles(joint_names, mu_est_hsmm[-1].tolist(), 1.0)
	pbd.plot_gmm3d(ax, model.mu[:,:3], model.sigma[:,:3,:3], color='blue', alpha=0.1)
	for i in range(model.nb_states):
		pbd.plot_gauss3d(ax, model.mu[i,:3], model.sigma[i,:3,:3],
					 n_points=20, n_rings=15, color='red', alpha=alpha_hsmm[i, -1])
	simulation_manager.stepSimulation(client_id)
	plt.pause(0.1)

plt.ioff()
plt.show()
cv2.destroyAllWindows()
simulation_manager.stopSimulation(client_id)