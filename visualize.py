import numpy as np 
from itertools import product, combinations, cycle
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from skimage.draw import ellipsoid
from pyntcloud import PyntCloud
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.lines import Line2D

def DrawConfidenceRegion(s,center,rotation,trocar):

	radii = 1/np.sqrt(s)
	print(rotation.shape)
	# center = [0,0,0]
	# now carry on with EOL's answer
	u = np.linspace(0.0, 2.0 * np.pi, 100)
	v = np.linspace(0.0, np.pi, 100)
	x = radii[0] * np.outer(np.cos(u), np.sin(v))
	y = radii[1] * np.outer(np.sin(u), np.sin(v))
	z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
	for i in range(len(x)):
	    for j in range(len(x)):
	        [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]],rotation) + center
	        # x[i,j] = temp[0,0]
	        # y[i,j] = temp[0,1]
	        # z[i,j] = temp[0,2]
	# make some purdy axes
	axes = np.array([[radii[0],0.0,0.0],
	                 [0.0,radii[1],0.0],
	                 [0.0,0.0,radii[2]]])
	# rotate accordingly
	for i in range(len(axes)):
	    axes[i] = np.dot(axes[i], rotation)

	# plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# plot axes
	for p in axes:
	    X3 = np.linspace(-p[0], p[0], 100) + center[0]
	    Y3 = np.linspace(-p[1], p[1], 100) + center[1]
	    Z3 = np.linspace(-p[2], p[2], 100) + center[2]
	    ax.plot(X3, Y3, Z3, color='b')
	ax.set(xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)')
	ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
	plt.title(trocar + " confidence region")
	plt.show()

def visualize_model(trocar=None, pts = None, vect_end = None, vect_start = None, line_idx = None, gt=True):

	# Draw 3d graph

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('X (mm)')
	ax.set_ylabel('Y (mm)')
	ax.set_zlabel('Z (mm)')

	# r = [-120, 120]
	# for s, e in combinations(np.array(list(product(r, r, r))), 2):
	# 	if np.sum(np.abs(s-e)) == r[1]-r[0]:
	# 		ax.plot3D(*zip(s, e), color="#0c0c0d")


	# draw cloud points
	if pts is not None:		
		for i in range(pts.shape[0]):
			ax.scatter(pts[i,0],pts[i,1],pts[i,2],marker = ",",color="#948e8e")

	if line_idx is not None:

		#draw lines in each cluster
		cycol = cycle('grcmy')
		# for i in range(len(line_idx)):
		# 	color = next(cycol)
		# 	for idx in line_idx[i]:
		# 		ax.plot([vect_start[idx][0], vect_end[idx][0]], [vect_start[idx][1],vect_end[idx][1]],zs=[vect_start[idx][2],vect_end[idx][2]],color=color)
		for key,value in line_idx.items():

			color = next(cycol)

			for val in value:

				ax.plot([vect_start[val][0], vect_end[val][0]], [vect_start[val][1],vect_end[val][1]],zs=[vect_start[val][2],vect_end[val][2]],color=color)

		custom_lines = [Line2D([0], [0], color='g', lw=2),
						Line2D([0], [0], color='r', lw=2),
						Line2D([0], [0], color='c', lw=2),
						Line2D([0], [0], color='m', lw=2),
						Line2D([0], [0], color='y', lw=2)]

		ax.legend(custom_lines, list(line_idx.keys()),loc='upper right')

	else:
		#draw all lines
		if vect_start is not None:
			N_lines = vect_start.shape[0]
			for i in range(N_lines):
			    ax.plot([vect_start[i][0], vect_end[i][0]], [vect_start[i][1],vect_end[i][1]],zs=[vect_start[i][2],vect_end[i][2]])


	#draw trocar point
	if trocar is not None:
		for i in range(trocar.shape[0]):
			ax.scatter(trocar[i,0],trocar[i,1],trocar[i,2],marker = "*",color="b")

	#draw original point O(0,0,0)
	ax.scatter(0,0,0,marker = "o",color='r')

	if gt:
		plt.title("Real Data")
	else:
		plt.title("Clustered Data")


	plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_cfs_matrix(y_true,y_pred,labels):

	# if include_noise:

	# 	cm = confusion_matrix(y_true,y_pred,labels=["Trocar 1","Trocar 2","Trocar 3","Trocar 4","Incorrect Data"],normalize='true')
	# 	disp = ConfusionMatrixDisplay(confusion_matrix=cm,
	# 	                      display_labels=["Trocar 1","Trocar 2","Trocar 3","Trocar 4","Incorrect Data"]).plot()
	# else:

	# 	cm = confusion_matrix(y_true,y_pred,labels=["Trocar 1","Trocar 2","Trocar 3","Trocar 4"],normalize='true')
	# 	disp = ConfusionMatrixDisplay(confusion_matrix=cm,
	# 	                      display_labels=["Trocar 1","Trocar 2","Trocar 3","Trocar 4"]).plot()

	cm = confusion_matrix(y_true,y_pred,labels=labels,normalize='true')
	disp = ConfusionMatrixDisplay(confusion_matrix=cm,
	                      display_labels=labels).plot()	
	# NOTE: Fill all variables here with default values of the plot_confusion_matrix
	plt.show()

