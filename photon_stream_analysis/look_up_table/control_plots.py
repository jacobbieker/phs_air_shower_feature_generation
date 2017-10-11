import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from tqdm import tqdm


def save_plot_cx_cy_source_position_in_camera(lut, out_path):
	PAP = []
	for i in tqdm(range(lut.number_events)):
		PAP.append(lut.pap(i))
	dfpap = pd.DataFrame(PAP)


	plt.figure()
	plt.hist2d(np.rad2deg(dfpap[2]), np.rad2deg(dfpap[3]), bins=42)
	plt.xlabel('cx/deg')
	plt.ylabel('cy/deg')
	ax = plt.gca()
	ax.set_aspect('equal')
	plt.savefig(out_path+'_cx_cy.png')

	plt.figure()
	plt.hist2d(dfpap[0], dfpap[1], bins=42)
	plt.xlabel('x/m')
	plt.ylabel('y/m')
	ax = plt.gca()
	ax.set_aspect('equal')
	plt.savefig(out_path+'_x_y.png')

	return dfpap