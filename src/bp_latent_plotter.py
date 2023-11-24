import numpy as np
# import matplotlib.pyplot as plt

import plotly.graph_objs as go
from plotly.subplots import make_subplots

latent_dataset = np.load('bp_latents_dataset.npz', allow_pickle=True)
latent_realtime = np.load('bp_latents_realtime.npz', allow_pickle=True)

fig = make_subplots(rows=1, cols=2,
					specs=[[{'is_3d': True} for i in range(2)]for v in range(1)],
					print_grid=False,
					subplot_titles=('Robot Latent Space', 'Human Dynamics Space')
					)

for zr_hri in latent_dataset['zr_hri']:
	s=np.linspace(1,10,len(zr_hri))
	for i in range(0, len(zr_hri), 5):
		fig.add_trace(go.Scatter3d(x=[zr_hri[i,0]], y=[zr_hri[i,1]], z=[zr_hri[i,2]],
                                   mode='markers',
								   marker=dict(
										size=s[i],
										color='red',
										opacity=min(0.7,max(0.1,(i+1)/len(zr_hri))),
									)
						),
						row=1, col=1,
					)

for dh in latent_dataset['dh']:
	s=np.linspace(1,10,len(dh))
	for i in range(0, len(dh), 5):
		fig.add_trace(go.Scatter3d(x=[dh[i,0]], y=[dh[i,1]], z=[dh[i,2]],
                                   mode='markers',
								   marker=dict(
										size=s[i],
										color='blue',
										opacity=min(0.7,max(0.1,(i+1)/len(dh))),
									)
						),
						row=1, col=2,
					)



zr_hri = latent_realtime['zr_hri']
s=np.linspace(1,10,len(zr_hri))
for i in range(0, len(zr_hri), 5):
	fig.add_trace(go.Scatter3d(x=[zr_hri[i,0]], y=[zr_hri[i,1]], z=[zr_hri[i,2]],
								mode='markers',
								marker=dict(
									size=s[i],
									color='magenta',
									opacity=min(0.7,max(0.1,(i+1)/len(zr_hri))),
								)
					),
					row=1, col=1,
				)

dh = latent_realtime['dh']
s=np.linspace(1,10,len(dh))
for i in range(0, len(dh), 5):
	fig.add_trace(go.Scatter3d(x=[dh[i,0]], y=[dh[i,1]], z=[dh[i,2]],
								mode='markers',
								marker=dict(
									size=s[i],
									color='cyan',
									opacity=min(0.7,max(0.1,(i+1)/len(dh))),
								)
					),
					row=1, col=2,
				)

fig.update_layout(height=900, width=1800)
fig.show()