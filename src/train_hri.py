import numpy as np
import pbdlib as pbd
import time
from utils import *

from qibullet import SimulationManager

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

data = np.load('data/labelled_sequences_prolonged.npz', allow_pickle=True)
print([k for k in data.keys()])

# hand_idx = np.array([27,28,29,57,58,59])
# train_data = [traj[:, None, hand_idx] for traj in data['train_data']]
# test_data = [traj[:, None, hand_idx] for traj in data['test_data']]

# # All joints
# train_data = [np.concatenate([traj[:, :30].reshape((-1,10,3)),traj[:, 30:].reshape((-1,10,3))], -1) for traj in data['train_data']]
# test_data = [np.concatenate([traj[:, :30].reshape((-1,10,3)),traj[:, 30:].reshape((-1,10,3))], -1) for traj in data['test_data']]

# Just hands
train_data = []
test_data = []
for traj in data['train_data']:
	hand_traj = traj[:, 27:30]
	joint_traj = np.array([joint_angle_extraction(skeleton) for skeleton in traj[:, 30:].reshape((-1, 10, 3))])
	train_data.append(np.concatenate([hand_traj, joint_traj], -1))

for traj in data['test_data']:
	hand_traj = traj[:, 27:30]
	joint_traj = np.array([joint_angle_extraction(skeleton) for skeleton in traj[:, 30:].reshape((-1, 10, 3))])
	test_data.append(np.concatenate([hand_traj, joint_traj], -1))

# train_trajs = [np.concatenate([traj[:,:3], np.diff(traj[:,:3], prepend=traj[0:1,:3], axis=0), traj[:,3:], np.diff(traj[:,3:], prepend=traj[0:1,3:], axis=0)], axis=-1) for traj in train_data]
# test_trajs = [np.concatenate([traj[:,:3], np.diff(traj[:,:3], prepend=traj[0:1,:3], axis=0), traj[:,3:], np.diff(traj[:,3:], prepend=traj[0:1,3:], axis=0)], axis=-1) for traj in test_data]

train_trajs = [np.concatenate([traj[:,:3], np.diff(traj[:,:3], prepend=traj[0:1,:3], axis=0), traj[:,3:]], axis=-1) for traj in train_data]
test_trajs = [np.concatenate([traj[:,:3], np.diff(traj[:,:3], prepend=traj[0:1,:3], axis=0), traj[:,3:]], axis=-1) for traj in test_data]

for x in train_data[0]:
	joint_angles = x[3:].tolist()
	pepper.setAngles(joint_names, joint_angles, 1.0)
	simulation_manager.stepSimulation(client_id)
	time.sleep(0.1)

for x in train_data[15]:
	joint_angles = x[3:].tolist()
	pepper.setAngles(joint_names, joint_angles, 1.0)
	simulation_manager.stepSimulation(client_id)
	time.sleep(0.1)

for idx in [0,15]:
	train_trajs_sample = train_trajs[idx:idx+15]
	model = pbd.HSMM(nb_dim=train_trajs_sample[0].shape[-1], nb_states=6)
	model.init_hmm_kbins(train_trajs_sample)
	model.em(train_trajs_sample)


	if idx == 0:
		np.save('models/handshake_hri_nojointvel.npy',model)
	elif idx == 15:
		np.save('models/rocket_hri_nojointvel.npy',model)