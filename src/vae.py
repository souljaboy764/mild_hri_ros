import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal, kl_divergence

_eps = 1e-8

class VAE(nn.Module):
	def __init__(self, **kwargs):
		super(VAE, self).__init__()
		for key in kwargs:
			setattr(self, key, kwargs[key])

		self.activation = getattr(nn, kwargs['activation'])()
		self.input_dim = self.num_joints * self.joint_dims * self.window_size
		
		self.enc_sizes = [self.input_dim] + self.hidden_sizes
		enc_layers = []
		for i in range(len(self.enc_sizes)-1):
			enc_layers.append(nn.Linear(self.enc_sizes[i], self.enc_sizes[i+1]))
			enc_layers.append(self.activation)
		self._encoder = nn.Sequential(*enc_layers)

		# Legacy from 2022. not used
		self.latent = nn.Linear(self.enc_sizes[-1], self.latent_dim)

		self.post_mean = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		self.post_logstd = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		
		self.dec_sizes = [self.latent_dim] + self.hidden_sizes[::-1]
		dec_layers = []
		for i in range(len(self.dec_sizes)-1):
			dec_layers.append(nn.Linear(self.dec_sizes[i], self.dec_sizes[i+1]))
			dec_layers.append(self.activation)
		self._decoder = nn.Sequential(*dec_layers)
		self._output = nn.Linear(self.dec_sizes[-1], self.input_dim)
		
		
	def forward(self, x, encode_only = False, dist_only=False):
		enc = self._encoder(x)
		z_mean = self.post_mean(enc)
		if encode_only:
			return z_mean
		z_std = self.post_logstd(enc).exp() + _eps
		if dist_only:
			return MultivariateNormal(z_mean, scale_tril=torch.diag_embed(z_std))
			
		if self.training:
			eps = torch.randn((self.mce_samples,)+z_mean.shape, device=z_mean.device)
			zpost_samples = z_mean + eps*z_std
			zpost_samples = torch.concat([zpost_samples, z_mean[None]], dim=0)
		else:
			zpost_samples = z_mean
		
		x_gen = self._output(self._decoder(zpost_samples))
		# return x_gen, zpost_samples, z_mean, torch.diag_embed(z_std**2)
		return x_gen, zpost_samples, MultivariateNormal(z_mean, torch.diag_embed(z_std**2))

	def latent_loss(self, zpost_dist, zpost_samples):
		return kl_divergence(zpost_dist, Normal(0, 1)).mean()

class FullCovVAE(VAE):
	def __init__(self, **kwargs):
		super(FullCovVAE, self).__init__(**kwargs)
		
		self.post_cholesky = nn.Linear(self.enc_sizes[-1], (self.latent_dim*(self.latent_dim+1))//2)
		self.diag_idx = torch.arange(self.latent_dim)
		self.tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=0)

	def forward(self, x, encode_only = False, dist_only = False):
		enc = self._encoder(x)
		
		z_mean = self.post_mean(enc)
		if encode_only:
			return z_mean
		# Colesky Matrix Prediction 
		# Dorta et al. "Structured Uncertainty Prediction Networks" CVPR'18
		# Dorta et al. "Training VAEs Under Structured Residuals" 2018
		z_std = self.post_cholesky(enc)
		z_chol = torch.zeros(z_std.shape[:-1]+(self.latent_dim, self.latent_dim), device=z_std.device, dtype=z_std.dtype)
		z_chol[..., self.tril_indices[0], self.tril_indices[1]] = z_std
		z_chol[..., self.diag_idx,self.diag_idx] = 2*torch.abs(z_chol[..., self.diag_idx,self.diag_idx]) + _eps
		zpost_dist = MultivariateNormal(z_mean, scale_tril=z_chol)
		if dist_only:
			return zpost_dist
			
		if self.training:
			zpost_samples = torch.concat([zpost_dist.rsample((self.mce_samples,)), z_mean[None]], dim=0)
		else:
			zpost_samples = z_mean
		
		x_gen = self._output(self._decoder(zpost_samples))
		return x_gen, zpost_samples, zpost_dist

	def latent_loss(self, zpost_dist, zpost_samples):
		if isinstance(self.z_prior, Normal):
			return kl_divergence(zpost_dist, self.z_prior).mean()
		if isinstance(self.z_prior, list):
			kld = []
			for p in self.z_prior:
				kld.append(kl_divergence(zpost_dist, p))
			return kld
