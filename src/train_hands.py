import numpy as np
import pbdlib as pbd

from utils.visualization import *

data = np.load('data/labelled_sequences_prolonged.npz', allow_pickle=True)
print([k for k in data.keys()])

# hand_idx = np.array([27,28,29,57,58,59])
# train_data = [traj[:, None, hand_idx] for traj in data['train_data']]
# test_data = [traj[:, None, hand_idx] for traj in data['test_data']]

# # All joints
# train_data = [np.concatenate([traj[:, :30].reshape((-1,10,3)),traj[:, 30:].reshape((-1,10,3))], -1) for traj in data['train_data']]
# test_data = [np.concatenate([traj[:, :30].reshape((-1,10,3)),traj[:, 30:].reshape((-1,10,3))], -1) for traj in data['test_data']]

# Just hands
train_data = [np.concatenate([traj[:, :30].reshape((-1,10,3)),traj[:, 30:].reshape((-1,10,3))], -1) for traj in data['train_data']]
test_data = [np.concatenate([traj[:, :30].reshape((-1,10,3)),traj[:, 30:].reshape((-1,10,3))], -1) for traj in data['test_data']]

print(len(train_data), train_data[0].shape)
for i in range(len(train_data)):
	train_data[i][:,:,3] = -0.3 - train_data[i][:,:,3]
	train_data[i][:,:,4] = 0.85 - train_data[i][:,:,4]

for i in range(len(test_data)):
	test_data[i][:,:,3] = -0.3 - test_data[i][:,:,3]
	test_data[i][:,:,4] = 0.85 - test_data[i][:,:,4]
# visualize_skeleton(train_data[0])
# visualize_skeleton(train_data[15])
shake_range = range(0,15)
rocket_range = range(15,30)
hand_trajs_train = [traj[:,-1,:] for traj in train_data]
hand_trajs_test = [traj[:,-1,:] for traj in test_data]

train_trajs = [np.concatenate([traj[:,:3], np.diff(traj[:,:3], prepend=traj[0:1,:3], axis=0), traj[:,3:], np.diff(traj[:,3:], prepend=traj[0:1,3:], axis=0)], axis=-1) for traj in hand_trajs_train]
test_trajs = [np.concatenate([traj[:,:3], np.diff(traj[:,:3], prepend=traj[0:1,:3], axis=0), traj[:,3:], np.diff(traj[:,3:], prepend=traj[0:1,3:], axis=0)], axis=-1) for traj in hand_trajs_test]

for idx in [0,15]:
	train_trajs_sample = train_trajs[idx:idx+15]
	model = pbd.HSMM(nb_dim=train_trajs_sample[0].shape[-1], nb_states=6)
	model.init_hmm_kbins(train_trajs_sample)
	model.em(train_trajs_sample)

	fig = plt.figure()
	ax = fig.add_subplot(1, 2, 1, projection='3d')

	for i in range(15):
		ax.scatter(train_trajs_sample[i][:, 0], train_trajs_sample[i][:, 1], train_trajs_sample[i][:, 2], color='b', marker='o', alpha=0.09)
		# ax.scatter(train_trajs_sample[i][:, 3], train_trajs_sample[i][:, 4], train_trajs_sample[i][:, 5], color='b', marker='o', alpha=0.09)
		ax.scatter(train_trajs_sample[i][:, 6], train_trajs_sample[i][:, 7], train_trajs_sample[i][:, 8], color='y', marker='o', alpha=0.09)
	
	pbd.plot_gmm3d(ax, model.mu[:,:3], model.sigma[:,:3,:3])
	# pbd.plot_gmm3d(ax, model.mu[:,3:6], model.sigma[:,3:6,3:6], color='green')
	pbd.plot_gmm3d(ax, model.mu[:,6:9], model.sigma[:,6:9,6:9], color='green')
	for k in range(len(model.mu)):
		ax.text(model.mu[k,0], model.mu[k,1], model.mu[k,2],  '%s' % (str(k)), size=25, zorder=1, color='k')
		# ax.text(model.mu[k,3], model.mu[k,4], model.mu[k,5],  '%s' % (str(k)), size=25, zorder=1, color='k')
		ax.text(model.mu[k,6], model.mu[k,7], model.mu[k,8],  '%s' % (str(k)), size=25, zorder=1, color='k')

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax = fig.add_subplot(1, 2, 2)
	ax.imshow(np.log(model.Trans+1e-10), interpolation='nearest', vmin=-5, cmap='viridis')
	plt.show()
	plt.close()

	fig = plt.figure()
	ax = fig.add_subplot(1, 2, 1, projection='3d')
	mu_est_hsmm, sigma_est_hsmm = model.condition(test_trajs[0][:,:6], dim_in=slice(0, 6), dim_out=slice(6, 12))
	ax.plot(test_trajs[0][:, 0], test_trajs[0][:, 1], test_trajs[0][:, 2], color='b', marker='o', alpha=0.2)
	ax.plot(test_trajs[0][:, 6], test_trajs[0][:, 7], test_trajs[0][:, 8], color='r', marker='o', alpha=0.2)
	ax.plot(mu_est_hsmm[:, 0], mu_est_hsmm[:, 1], mu_est_hsmm[:, 2], color='y', marker='o', alpha=0.2)
	pbd.plot_gmm3d(ax, model.mu[:,6:9], model.sigma[:,6:9,6:9], color='green', alpha=0.2)
	
	ax = fig.add_subplot(1, 2, 2)
	
	alpha_hsmm = model.forward_variable(len(test_trajs[0][:,:6]), test_trajs[0][:,:6], slice(0, 6))
	ax.plot(alpha_hsmm.T)
	plt.show()
	# if idx == 0:
	# 	np.save('models_test/handshake_hands.npy',model)
	# elif idx == 15:
	# 	np.save('models_test/rocket_hands.npy',model)