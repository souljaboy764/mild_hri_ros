import numpy as np
import pbdlib as pbd

def fwd_init(model, nbD, priors):
	"""

	:param nbD:
	:return:
	"""
	bmx = np.zeros((model.nb_states, 1))

	Btmp = priors
	ALPHA = np.tile(model.init_priors, [nbD, 1]).T * model.Pd

	# r = Btmp.T * np.sum(ALPHA, axis=1)
	r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))

	bmx[:, 0] = Btmp / r
	E = bmx * ALPHA[:, [0]]
	S = np.dot(model.Trans_Pd.T, E)  # use [idx] to keep the dimension

	return bmx, ALPHA, S, Btmp * np.sum(ALPHA, axis=1)

def fwd_step(model, bmx, ALPHA, S, nbD, obs_marginal=None):
	"""

	:param bmx:
	:param ALPHA:
	:param S:
	:param nbD:
	:return:
	"""

	Btmp = obs_marginal

	ALPHA = np.concatenate((S[:, [-1]] * model.Pd[:, 0:nbD - 1] + bmx[:, [-1]] * ALPHA[:, 1:nbD],
							S[:, [-1]] * model.Pd[:, [nbD - 1]]), axis=1)

	r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))
	bmx = np.concatenate((bmx, Btmp[:, None] / r), axis=1)
	E = bmx[:, [-1]] * ALPHA[:, [0]]

	S = np.concatenate((S, np.dot(model.Trans_Pd.T, ALPHA[:, [0]])), axis=1)
	alpha = Btmp * np.sum(ALPHA, axis=1) + 1e-8
	alpha /= np.sum(alpha)
	return bmx, ALPHA, S, alpha


def forward_variable(model, n_step=None, demo=None, marginal=None, dep=None, p_obs=None):
	"""
	Compute the forward variable with some observations

	:param demo: 	[np.array([nb_timestep, nb_dim])]
	:param dep: 	[A x [B x [int]]] A list of list of dimensions
		Each list of dimensions indicates a dependence of variables in the covariance matrix
		E.g. [[0],[1],[2]] indicates a diagonal covariance matrix
		E.g. [[0, 1], [2]] indicates a full covariance matrix between [0, 1] and no
		covariance with dim [2]
	:param table: 	np.array([nb_states, nb_demos]) - composed of 0 and 1
		A mask that avoid some demos to be assigned to some states
	:param marginal: [slice(dim_start, dim_end)] or []
		If not None, compute messages with marginals probabilities
		If [] compute messages without observations, use size
		(can be used for time-series regression)
	:param p_obs: 		np.array([nb_states, nb_timesteps])
			custom observation probabilities
	:return:
	"""
	if isinstance(demo, np.ndarray):
		n_step = demo.shape[0]
	elif isinstance(demo, dict):
		n_step = demo['x'].shape[0]

	nbD = np.round(4 * n_step // model.nb_states)
	if nbD == 0:
		nbD = 10
	model.Pd = np.zeros((model.nb_states, nbD))
	# Precomputation of duration probabilities
	for i in range(model.nb_states):
		model.Pd[i, :] = pbd.multi_variate_normal(np.arange(nbD), model.Mu_Pd[i], model.Sigma_Pd[i], log=False)
		model.Pd[i, :] = model.Pd[i, :] / (np.sum(model.Pd[i, :])+pbd.realmin)
	# compute observation marginal probabilities
	p_obs, _ = model.obs_likelihood(demo, dep, marginal, n_step)

	model._B = p_obs

	h = np.zeros((model.nb_states, n_step))
	bmx, ALPHA, S, h[:, 0] = fwd_init(model, nbD, p_obs[:, 0])

	for i in range(1, n_step):
		bmx, ALPHA, S, h[:, i] = fwd_step(model, bmx, ALPHA, S, nbD, p_obs[:, i])
	h += 1e-8
	h /= np.sum(h, axis=0)

	return h