import numpy as np

from ikpy.chain import Chain
from ikpy.utils import plot

import numpy as np
import pbdlib as pbd

from utils.visualization import *
import matplotlib.pyplot as plt

data = np.load('data/labelled_sequences_prolonged.npz', allow_pickle=True)
print([k for k in data.keys()])

# Just hands
train_data = [np.concatenate([traj[:, :30].reshape((-1,10,3)),traj[:, 30:].reshape((-1,10,3))], -1) for traj in data['train_data']]
test_data = [np.concatenate([traj[:, :30].reshape((-1,10,3)),traj[:, 30:].reshape((-1,10,3))], -1) for traj in data['test_data']]

train_labels = data['train_labels']
test_labels = data['test_labels']
print(len(train_data), train_data[0].shape)
for i in range(len(train_data)):
	train_data[i][:,:,3] = -0.3 - train_data[i][:,:,3]
	train_data[i][:,:,4] = 0.85 - train_data[i][:,:,4]

for i in range(len(test_data)):
	test_data[i][:,:,3] = -0.3 - test_data[i][:,:,3]
	test_data[i][:,:,4] = 0.85 - test_data[i][:,:,4]

hand_trajs_train = [traj[:,-1,] for traj in train_data]
hand_trajs_test = [traj[:,-1,:] for traj in test_data]

for i in range(len(hand_trajs_train)):
    x1, y1, z1, x2, y2, z2 = np.array(hand_trajs_train[i]).T
    hand_trajs_train[i] = np.array([y1, -x1, z1, y2, -x2, z2]).T
for i in range(len(hand_trajs_test)):
    x1, y1, z1, x2, y2, z2 = np.array(hand_trajs_test[i]).T
    hand_trajs_test[i] = np.array([y1, -x1, z1, y2, -x2, z2]).T

train_trajs = [np.concatenate([traj[:,:3], np.diff(traj[:,:3], prepend=traj[0:1,:3], axis=0), traj[:,3:], np.diff(traj[:,3:], prepend=traj[0:1,3:], axis=0)], axis=-1) for traj in hand_trajs_train]
test_trajs = [np.concatenate([traj[:,:3], np.diff(traj[:,:3], prepend=traj[0:1,:3], axis=0), traj[:,3:], np.diff(traj[:,3:], prepend=traj[0:1,3:], axis=0)], axis=-1) for traj in hand_trajs_test]

idx = 0

train_trajs_sample = train_trajs[idx:idx+15]
print('train_trajs_sample.shape',train_trajs_sample[0].shape)
model = pbd.HSMM(nb_dim=train_trajs_sample[0].shape[-1], nb_states=6)
model.init_hmm_kbins(train_trajs_sample)
model.em(train_trajs_sample)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
mu_est_hsmm, sigma_est_hsmm = model.condition(train_trajs[idx][:,6:12], dim_in=slice(6, 12), dim_out=slice(0, 3))
alpha_hsmm = model.forward_variable(len(train_trajs[idx]), train_trajs[idx][:,6:12], slice(6, 12))
traj = np.array([mu_est_hsmm[:, 0]*2/5, mu_est_hsmm[:, 1]-0.1, (mu_est_hsmm[:, 2]+0.1)*4/5]).T
print(traj.shape)

pepper_left_arm_chain = Chain.from_json_file("resources/pepper/pepper_left_arm.json")
pepper_right_arm_chain = Chain.from_json_file("resources/pepper/pepper_right_arm.json")
pepper_legs_chain = Chain.from_json_file("resources/pepper/pepper_legs.json")
pepper_head_chain = Chain.from_json_file("resources/pepper/pepper_head.json")

pepper_head_chain.name = pepper_left_arm_chain.name = pepper_right_arm_chain.name = None

# fig, ax = plot.init_3d_figure()
# ax.set_xlim3d(-0.2, 0.6)
# ax.set_ylim3d(-0.4, 0.4)
# ax.set_zlim3d(-0.3, 0.5)
plt.ion()
plt.pause(0.1)
input()

for i in range(len(traj)):
    ax.clear()
    ax.set_xlim3d(-0.1, 0.9)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.6, 0.4)
    for j in range(len(model.mu)):
        pbd.plot_gauss3d(ax, model.mu[j,6:9], model.sigma[j,6:9,6:9],
					 color='red', alpha=min(alpha_hsmm[j, i] + 0.2,1))
        pbd.plot_gauss3d(ax, model.mu[j,:3], model.sigma[j,:3,:3],
					 color='blue', alpha=min(alpha_hsmm[j, i] + 0.2,1))
    
    frame_target = np.eye(4)
    frame_target[:3, 3] = traj[i]

    ik = pepper_right_arm_chain.inverse_kinematics_frame(frame_target)
    pepper_right_arm_chain.plot(ik, ax)
    
    pepper_left_arm_chain.plot([0, 0, np.pi/2, 0.009, 0, -0.009, 0, 0, 0], ax)
    # pepper_legs_chain.plot([0] * (len(pepper_legs_chain)), ax)
    pepper_head_chain.plot([0] * (len(pepper_head_chain)), ax)
    ax.plot(train_trajs[idx][i, 0], train_trajs[idx][i, 1], train_trajs[idx][i, 2], color='b', marker='o', label='Controlled Agent GT')
    ax.plot(train_trajs[idx][i, 6], train_trajs[idx][i, 7], train_trajs[idx][i, 8], color='r', marker='o', label='Observed Agent')
    ax.plot(mu_est_hsmm[i, 0], mu_est_hsmm[i, 1], mu_est_hsmm[i, 2], color='g', marker='o', label='Controlled Agent Pred')
    ax.plot(traj[i, 0], traj[i, 1], traj[i, 2], color='k', marker='o', label='Controlled Agent Pred Scaled')
    ax.legend()
    plt.pause(0.1)
plt.ioff()
plt.show()
