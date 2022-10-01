from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np
import pbdlib as pbd

from utils import *

class Arrow3D(FancyArrowPatch):

	def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
		super().__init__((0, 0), (0, 0), *args, **kwargs)
		self._xyz = (x, y, z)
		self._dxdydz = (dx, dy, dz)

	def draw(self, renderer):
		x1, y1, z1 = self._xyz
		dx, dy, dz = self._dxdydz
		x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

		xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
		self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
		super().draw(renderer)
		
	def do_3d_projection(self, renderer=None):
		x1, y1, z1 = self._xyz
		dx, dy, dz = self._dxdydz
		x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

		xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
		self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

		return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
	'''Add an 3d arrow to an `Axes3D` instance.'''

	arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
	ax.add_artist(arrow)


setattr(Axes, 'arrow3D', _arrow3D)
point_color = (0, 255, 0)
line_color = (0, 0, 225)


def visualize_skeleton(data, variant=None, action=None):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	# plt.ion()

	print('data',data.shape)
	# center_mean = (data[0, :, :3].sum(axis=0) + data[0, :, 3:].sum(axis=0))/(2*data.shape[1])
	# print(center_mean)
	# data[:, :, :3] = data[:, :, :3] - center_mean
	# data[:, :, 3:] = data[:, :, 3:] - center_mean
	ax.view_init(0, -0)
	# ax.grid(False)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	# ax.set_axis_bgcolor('white')w
	for frame_idx in range(data.shape[0]):
		ax.cla()
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.set_facecolor('none')
		ax.set_xlim3d([-0.9, 0.1])
		ax.set_ylim3d([-0.1, 0.9])
		ax.set_zlim3d([-0.65, 0.35])
		ax.set_title("Frame: {}".format(frame_idx))

		# ax.axis('off')
		if variant is not None and action is not None:
			# ax.set_title('_'.join(variant.split('/')[0]) + " " + action)
			ax.set_title(variant + " " + action)

		x = data[frame_idx, :, 0]
		y = data[frame_idx, :, 1]
		z = data[frame_idx, :, 2]
		ax.scatter(x, y, z, color='r', marker='o')

		x = data[frame_idx, :, 3]
		y = data[frame_idx, :, 4]
		z = data[frame_idx, :, 5]
		ax.scatter(x, y, z, color='b', marker='o')
		plt.pause(0.01)
		if not plt.fignum_exists(1):
			break
	
	plt.ioff()
	plt.show()

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

def plot_pbd(ax, model, alpha_hsmm=None):
	pbd.plot_gmm3d(ax, model.mu[:,:3], model.sigma[:,:3,:3], color='blue', alpha=0.1)
	if alpha_hsmm is not None:
		for i in range(model.nb_states):
			pbd.plot_gauss3d(ax, model.mu[i,:3], model.sigma[i,:3,:3],
						n_points=20, n_rings=15, color='red', alpha=alpha_hsmm[i, -1])
