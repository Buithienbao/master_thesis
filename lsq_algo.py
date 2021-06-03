import numpy as np 
import shapely
from generate_data import *
from visualize import *
from scipy.optimize import least_squares,leastsq
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import math
from sklearn.utils import shuffle
from skimage.draw import ellipsoid
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,mean_absolute_error
from pyntcloud import PyntCloud
from plyfile import PlyData
from itertools import product, combinations, cycle
import time
import statsmodels.api as sm
import pickle
import os 
import glob
from scipy.io import loadmat

N_lines = 1000
# percentage = 0.2
# num_outliers = int(N_lines*percentage)

start_range = 0
end_range = 50
# step = (end_range - start_range)/10
step = 5
SCALE_COEF1 = 10
SCALE_COEF2 = 5

def lineseg_dist(p, a, b, index = 0, list_idx_lines = [], threshold = 0):

	dist = []
	dist_val = 0
	# list_idx = []
	# temp_dist = []
	count = 0

	if len(list_idx_lines):

		for i in range(len(list_idx_lines)):
			x = a[list_idx_lines[i]] - b[list_idx_lines[i]]
			t = np.dot(p-b[list_idx_lines[i]], x)/np.dot(x, x)
			dist_val = np.linalg.norm(t*(a[list_idx_lines[i]]-b[list_idx_lines[i]])+b[list_idx_lines[i]]-p)
			dist.append(dist_val)

	else:

		for i in range(len(a)):

			x = a[i] - b[i]
			t = np.dot(p-b[i], x)/np.dot(x, x)
			dist_val = np.linalg.norm(t*(a[i]-b[i])+b[i]-p)
			dist.append(dist_val)
	
	if index:
		
		# list_idx = np.zeros(index,dtype=np.uint8)
		temp_dist = dist.copy()
		temp_dist = sorted(temp_dist)
		temp_dist = temp_dist[0:index]

		# max_dist = max(dist)
		# print(dist)
		# for j in range(index):
			# count += 1
		list_idx = sorted(range(len(dist)), key=lambda k: dist[k])[:index]
		# temp_dist = np.asarray(temp_dist,dtype=np.uint8)
			# list_idx.append(list_idx_lines[temp_idx])
			# list_idx[j] = list_idx_lines[temp_idx]
			# dist[temp_idx] = max_dist
		# print(count)
		# print(len(temp_dist))
		# print(len(np.unique(list_idx)))
		# print(len(np.unique(list_idx)))
		# print(len(np.unique(list_idx)) == len(list_idx))
		final_dist = np.linalg.norm(temp_dist)
		# print(final_dist)
		return final_dist, list_idx_lines[list_idx]

	if threshold:

		temp_dist = np.array(dist.copy())
		list_idx = np.where(temp_dist < threshold)
		return list_idx_lines[list_idx]

	return np.linalg.norm(dist)

def relative_err_calc(pred,gt):

	return np.abs((pred-gt)/gt)*100

def abs_err_calc(pred,gt):

	return np.abs(pred-gt)

def eudist_err_calc(pred,gt):

	return np.linalg.norm(pred-gt)

def linear_least_squares(a, b, residuals=False):
    """
    Return the least-squares solution to a linear matrix equation.
    Solves the equation `a x = b` by computing a vector `x` that
    minimizes the Euclidean 2-norm `|| b - a x ||^2`.  The equation may
    be under-, well-, or over- determined (i.e., the number of
    linearly independent rows of `a` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `a`
    is square and of full rank, then `x` (but for round-off error) is
    the "exact" solution of the equation.
    Parameters
    ----------
    a : (M, N) array_like
        "Coefficient" matrix.
    b : (M,) array_like
        Ordinate or "dependent variable" values.
    residuals : bool
        Compute the residuals associated with the least-squares solution
    Returns
    -------
    x : (M,) ndarray
        Least-squares solution. The shape of `x` depends on the shape of
        `b`.
    residuals : int (Optional)
        Sums of residuals; squared Euclidean 2-norm for each column in
        ``b - a*x``.
    """
    if type(a) != np.ndarray or not a.flags['C_CONTIGUOUS']:
        warn('Matrix a is not a C-contiguous numpy array. The solver will create a copy, which will result' + \
             ' in increased memory usage.')

    a = np.asarray(a, order='c')
    i = dgemm(alpha=1.0, a=a.T, b=a.T, trans_b=True)
    x = np.linalg.solve(i, dgemm(alpha=1.0, a=a.T, b=b)).flatten()

    if residuals:
        return x, np.linalg.norm(np.dot(a, x) - b)
    else:
        return x

def find_intersection_3d_lines(p1,p2,p3,p4):

	"""
	Find intersection 3d lines

	Parameters
	----------
	p1,p2,p3,p4 : numpy.ndarray
		coordinate of 3D points (p1 p2 lies on line 1, p3 p4 lies on line 2), an array of size (3,)

	Returns
	-------
	p_intsec : numpy.ndarray
		coordinate of 3D intersection point, an array of size (3,)

	"""
	a1 = np.dot(p1-p3,p4-p3)*np.dot(p4-p3,p2-p1) - np.dot(p1-p3,p2-p1)*np.dot(p4-p3,p4-p3)
	b1 = np.dot(p2-p1,p2-p1)*np.dot(p4-p3,p4-p3) - np.dot(p4-p3,p2-p1)*np.dot(p4-p3,p2-p1)

	if b1:
		coef1 = a1/b1    	
	else:
		return np.array([])

	a2 = np.dot(p1-p3,p4-p3) + coef1*np.dot(p4-p3,p2-p1)
	b2 = np.dot(p4-p3,p4-p3)

	if b2:
		coef2 = a2/b2
	else:
		return np.array([])

	pt1 = p1 + coef1*(p2-p1)
	pt2 = p3 + coef2*(p4-p3)

	diff = np.linalg.norm(pt1 - pt2)

	if diff < 0.001:

		return pt1

	else:

		return (pt1+pt2)/2
	

def EvaluateLsqSolution(covariance_mtrx, pts_estimated):
	'''
	Evaluate the estimation result using linearized statistics
	'''

def test_case(trocar, percentage, choice,N_lines = 1000, sigma=5, upper_bound=150):

	lst = ['incorrect_data','noise','observed lines']
	data_path = '/home/bao/Downloads/Git/master_thesis/data'
	num_trocar = trocar.shape[0]

	if choice == lst[0]:
		list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
		pref = 'inc'
		pref_err = 'inc_err'
	elif choice == lst[1]:
		list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
		pref = 'sigma'
		pref_err = 'sigma_err'

	else:
		list_noise_percentage = np.arange(start_range+step, end_range + step,step,dtype=np.uint8)
		list_noise_percentage = [element * 20 for element in list_noise_percentage]
		pref = 'lines'
		pref_err = 'lines_err'

	path = os.path.join(data_path,pref)

	lst_data = glob.glob(path + '/' + "*.npz")
	lst_data.sort()

	lst_gt = glob.glob(path + '/' + "*.pkl")
	lst_gt.sort()

	path_err = os.path.join(data_path,pref_err)

	lst_err = glob.glob(path_err + '/' + "*.npz")
	lst_err.sort()

	list_abs_err = np.zeros((len(list_noise_percentage),num_trocar),dtype=np.float32)
	# list_acc = np.zeros((len(list_noise_percentage),num_trocar),dtype=np.float32)
	list_acc = np.zeros((len(list_noise_percentage),1),dtype=np.float32)
	list_trocar = np.zeros((len(list_noise_percentage),1),dtype=np.uint8)

	ite = 0

	for num in list_noise_percentage:

		if choice == lst[0]:
			percentage[-1] = num/100
		elif choice ==lst[1]:
			sigma = num
		else:
			N_lines = num
		# vect_start, vect_end, dict_gt = generate_data(N_lines=N_lines, percentage = percentage, trocar=trocar, scale1 = SCALE_COEF1, scale2 = SCALE_COEF2, sigma = sigma, upper_bound = upper_bound)
		# print(lst_data[ite])
		# print(lst_gt[ite])
		vect_start,vect_end,dict_gt = load_dataset(lst_data[ite],lst_gt[ite])

		# print(dict_gt)
		abs_err, acc_clustering, num_trocar = ransac_new(trocar, vect_start, vect_end, dict_gt, N_lines = N_lines)

		# for i in range(len(acc_clustering)):

		# 	list_acc[ite,i] = acc_clustering[i]*100

		list_acc[ite] = acc_clustering*100
		list_trocar[ite] = num_trocar

		for i in range(len(abs_err)):

			list_abs_err[ite,i] = abs_err[i]

		# acc_clustering,num_trocar = ransac_new(trocar,percentage,N_lines,sigma,upper_bound,test_num_clus=True)
		# acc_clustering,num_trocar = ransac_new(trocar, vect_start, vect_end, dict_gt, N_lines = N_lines, test_num_clus=True)

		# list_trocar[ite] = num_trocar
		# list_acc[ite] = acc_clustering*100

		ite += 1

	list_pos = np.copy(list_abs_err)
	list_pos[list_pos < 0 ] = np.nan
	
	if lst_err:
		
		data = np.load(lst_err[0])
		trocar1 = data['trocar1']
		trocar2 = data['trocar2']
		trocar3 = data['trocar3']
		trocar4 = data['trocar4']
		mean_err = data['mean_err']
		acc = data['acc']
		nt = data['num_trocar']
		trocar1	= np.append(trocar1,list_abs_err[:,0],axis=0)
		trocar2	= np.append(trocar2,list_abs_err[:,1],axis=0)
		trocar3	= np.append(trocar3,list_abs_err[:,2],axis=0)
		trocar4	= np.append(trocar4,list_abs_err[:,3],axis=0)
		mean_err = np.append(mean_err,np.nanmean(list_pos,axis=1),axis=0)
		acc	= np.append(acc,list_acc,axis=0)
		nt	= np.append(nt,list_trocar,axis=0)

		np.savez(os.path.join(path_err,pref_err+'.npz'), trocar1=trocar1, trocar2=trocar2, trocar3=trocar3, trocar4=trocar4, mean_err=mean_err, acc=acc,num_trocar=nt)

	else:

		np.savez(os.path.join(path_err,pref_err+'.npz'), trocar1=list_abs_err[:,0], trocar2=list_abs_err[:,1], trocar3=list_abs_err[:,2], trocar4=list_abs_err[:,3], mean_err=np.nanmean(list_pos,axis=1), acc=list_acc,num_trocar=list_trocar)
	

	# #plot the result

	# if choice == lst[1]:

	# 	plt.figure(100),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	axs.plot(list_noise_percentage, list_acc, 'o-')
	# 	axs.xaxis.grid(True, which='major')
	# 	axs.yaxis.grid(True, which='major')
	# 	axs.set(xlabel='Amount of noise (mm)', ylabel='Clustering accuracy (%)')

	# 	# fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	# axs.plot(list_noise_percentage, list_acc[:,0], 'r-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,1], 'b-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,2], 'g-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,3], 'm-')
	# 	# axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# 	# axs.set(xlabel='Amount of noise (mm)', ylabel='Clustering accuracy (%)')


	# 	plt.figure(200),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	axs.plot(list_noise_percentage, list_trocar, 'o-')
	# 	axs.xaxis.grid(True, which='major')
	# 	axs.yaxis.grid(True, which='major')
	# 	axs.set(xlabel='Amount of noise (mm)', ylabel='Number of trocar predicted')

	# 	plt.figure(300),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	axs.plot(list_noise_percentage, list_abs_err[:,0], 'o-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,1], 'o-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,2], 'o-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,3], 'o-')
	# 	axs.plot(list_noise_percentage, np.nanmean(list_pos,axis=1), 'o-')
	# 	axs.xaxis.grid(True, which='major')
	# 	axs.yaxis.grid(True, which='major')
	# 	axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4','Average'])
	# 	axs.set(xlabel='Amount of noise (mm)', ylabel='Trocar position error (mm)')

	# elif choice == lst[0]:

	# 	plt.figure(100),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	axs.plot(list_noise_percentage, list_acc, 'o-')
	# 	axs.xaxis.grid(True, which='major')
	# 	axs.yaxis.grid(True, which='major')
	# 	axs.set(xlabel='Incorrect data (%)', ylabel='Clustering accuracy (%)')

	# 	# fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	# axs.plot(list_noise_percentage, list_acc[:,0], 'r-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,1], 'b-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,2], 'g-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,3], 'm-')
	# 	# axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# 	# axs.set(xlabel='Incorrect data (%)', ylabel='Clustering accuracy (%)')

	# 	plt.figure(200),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	axs.plot(list_noise_percentage, list_trocar, 'o-')
	# 	axs.xaxis.grid(True, which='major')
	# 	axs.yaxis.grid(True, which='major')
	# 	axs.set(xlabel='Incorrect data (%)', ylabel='Number of trocar predicted')

	# 	plt.figure(300),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	axs.plot(list_noise_percentage, list_abs_err[:,0], 'o-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,1], 'o-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,2], 'o-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,3], 'o-')
	# 	axs.plot(list_noise_percentage, np.nanmean(list_pos,axis=1), 'o-')
	# 	axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4','Average'])
	# 	axs.xaxis.grid(True, which='major')
	# 	axs.yaxis.grid(True, which='major')
	# 	axs.set(xlabel='Incorrect data (%)', ylabel='Trocar position error (mm)')

	# else:
	# 	plt.figure(100),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	axs.plot(list_noise_percentage, list_acc, 'o-')
	# 	axs.xaxis.grid(True, which='major')
	# 	axs.yaxis.grid(True, which='major')
	# 	axs.set(xlabel='Number of observed tool 3D axes', ylabel='Clustering accuracy (%)')

	# 	# fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	# axs.plot(list_noise_percentage, list_acc[:,0], 'r-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,1], 'b-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,2], 'g-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,3], 'm-')
	# 	# axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# 	# axs.set(xlabel='Incorrect data (%)', ylabel='Clustering accuracy (%)')

	# 	plt.figure(200),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	axs.plot(list_noise_percentage, list_trocar, 'o-')
	# 	axs.xaxis.grid(True, which='major')
	# 	axs.yaxis.grid(True, which='major')
	# 	axs.set(xlabel='Number of observed tool 3D axes', ylabel='Number of trocar predicted')

	# 	plt.figure(300),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	axs.plot(list_noise_percentage, list_abs_err[:,0], 'o-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,1], 'o-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,2], 'o-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,3], 'o-')
	# 	axs.plot(list_noise_percentage, np.nanmean(list_pos,axis=1), 'o-')
	# 	axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4','Average'])
	# 	# axs.grid(True,linestyle='--')
	# 	axs.xaxis.grid(True, which='major')
	# 	axs.yaxis.grid(True, which='major')
	# 	axs.set(xlabel='Number of observed tool 3D axes', ylabel='Trocar position error (mm)')

	# plt.show()	



def ransac_new(trocar, vect_start, vect_end, dict_gt, N_lines = 1000):

	num_trocar = trocar.shape[0]

	dict_cluster = {}

	list_abs_err = np.zeros((num_trocar,1),dtype=np.float32)
	list_acc = np.zeros((num_trocar,1),dtype=np.float32)

	bool_continue = False

	while bool_continue == False:

		num_trials = 100000000
		sample_count = 0
		sample_size = 2
		P_min = 0.99
		temp_per = 0

		list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
		# remove_idx = []
		vect_clustered = []
		threshold_dist = 8
		threshold_inliers = 30
		vect_cent = []
		flat_list = []
		# start_time = time.time()
		count_cluster = 0

		while(num_trials > sample_count):
			
			sample_count += 1
			
			list_idx_copy = list_idx.copy()
			
			idx1 = random.choice(list_idx)
			# list_idx = np.delete(list_idx,np.where(list_idx==idx1))

			idx2 = random.choice(list_idx)
			
			# while(idx2 == idx1):
				
			# 	idx2 = random.choice(list_idx)
			# list_idx = np.delete(list_idx,np.where(list_idx==idx2))

			estim_pt = find_intersection_3d_lines(vect_end[idx1], vect_start[idx1], vect_end[idx2], vect_start[idx2])
			
			if not estim_pt.any():

				continue

			min_list_idx_temp = lineseg_dist(estim_pt, vect_start, vect_end, list_idx_lines = list_idx_copy, threshold = threshold_dist)

			num_inliers = len(min_list_idx_temp)
			# print("1st: ",num_inliers)

			if num_inliers < 2:

				continue

			center_point_temp,_,_,_ = estimate_trocar(vect_end,vect_start,min_list_idx_temp)

			min_list_idx_temp = lineseg_dist(center_point_temp,vect_start,vect_end, list_idx_lines = list_idx_copy, threshold = threshold_dist)

			num_inliers = len(min_list_idx_temp)

			# print("2nd: ",num_inliers)

			if num_inliers < 2:

				continue

			#update RANSAC params

			P_outlier = 1 - num_inliers/(N_lines-temp_per)
			
			if not P_outlier:

				vect_clustered.append(list_idx.tolist())
				list_idx = []
				break

			num_trials = int(math.log(1-P_min)/math.log(1-(1-P_outlier)**sample_size))


			if num_inliers > threshold_inliers:

				# # remove_idx.append(min_list_idx_temp)
				# center_point_temp,_,_,_ = estimate_trocar(vect_end,vect_start,min_list_idx_temp)

				# new_min_list_idx_temp = lineseg_dist(center_point_temp,vect_start,vect_end, list_idx_lines = list_idx_copy, threshold = threshold_dist)

				# new_num_inliers = len(new_min_list_idx_temp)

				# while new_num_inliers > num_inliers:

				# 	num_inliers = new_num_inliers

				# 	min_list_idx_temp = np.copy(new_min_list_idx_temp)

				# 	center_point_temp,_,_,_ = estimate_trocar(vect_end,vect_start,min_list_idx_temp)

				# 	new_min_list_idx_temp = lineseg_dist(center_point_temp,vect_start,vect_end, list_idx_lines = list_idx_copy, threshold = threshold_dist)

				# 	new_num_inliers = len(new_min_list_idx_temp)

				count_cluster+=1
				# print("Trocar " + str(count_cluster) + " found.")
				vect_cent.append(center_point_temp)
				vect_clustered.append(min_list_idx_temp.tolist())
				list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
				flat_list = [item for sublist in vect_clustered for item in sublist]
				flat_list = np.array(flat_list)
				# print(sorted(np.unique(flat_list)))
				list_idx = list_idx[~np.isin(list_idx,flat_list)]
				
				if not len(list_idx):
					# print("Last cluster. Length: 0")
					break

				elif len(list_idx) < 3:
					# print("Last cluster. Length: ",len(list_idx))
					vect_clustered.append(list_idx.tolist())
					list_idx = []
					break

				list_idx = shuffle(list_idx)

				#reset RANSAC params
				sample_count = 0
				num_trials = 100000000

				temp_per += num_inliers
				# print(num_inliers)

		#Store the last cluster (if any)
		if len(list_idx):
			# print("Last cluster. Length: ",len(list_idx))
			vect_clustered.append(list_idx.tolist())
	
		if len(vect_cent):
			
			bool_continue = True

	# print("--- %s seconds ---" % (time.time() - start_time))
	# print(vect_cent)
	y_true,y_pred,center_pts,dict_abs_err = flattenClusteringResult(dict_gt,dict_cluster,vect_clustered,vect_cent,trocar,N_lines)
	# plot_cfs_matrix(y_true,y_pred,list(dict_gt.keys()))
	# acc_clustering = confusion_matrix(y_true,y_pred,list(dict_gt.keys()), normalize="true").diagonal()
	acc_clustering = accuracy_score(y_true,y_pred)
	# print(center_pts)
	
	# if not test_num_clus:

	ite = 0

	for key,value in dict_abs_err.items():

		if ite == num_trocar:

			break

		# final_sol = center_pts[key]

		# abs_err = mean_absolute_error(final_sol,trocar[ite])
		
		list_abs_err[ite] = dict_abs_err[key]

		ite += 1

	# for i in range(acc_clustering.shape[0]):

	# 	if i == num_trocar:

	# 		break

	# 	list_acc[i] = acc_clustering[i]



	# pts = read_ply("liver_simplified.ply")


	# visualize_model(pts=pts,trocar=trocar,vect_end=vect_end,vect_start=vect_start,line_idx=dict_gt,gt=True)
	
	# visualize_model(trocar=trocar,vect_end=vect_end,vect_start=vect_start,line_idx=dict_cluster,gt=False)
		# visualize_model(pts=pts)

	return list_abs_err, acc_clustering, len(vect_cent)

	# else:

		# return acc_clustering,len(vect_clustered)

def flattenClusteringResult(dict_gt,dict_cluster,vect_clustered,vect_cent,trocar,N_lines):

	"""
	Convert from dictionary to 1D numpy array
	"""
	dict_cent = {}
	dict_abs_err = {}

	trocar_index = 0
	
	diff = 10000000
	diff1 = 10000000

	list_match_idx = []

	for key,value in dict_gt.items():

		match_max = 10000000000
		match_index = 0
		


		if diff1 == 0 and diff > 0:

			temp = []

			for i in range(len(vect_clustered)):

				if i in list_match_idx:

					continue

				temp.append(vect_clustered[i])

			flat_list = [item for sublist in temp for item in sublist]

			dict_cluster[key]  = flat_list
			
			dict_cent[key] = np.array([0,0,0])

			break

		elif diff == 0:

			dict_cent[key] = np.array([0,0,0])
			
			abs_err = -1
			
			# if trocar_index > trocar.shape[0] - 1:

			# 	abs_err = 0

			# else:
				
			# 	abs_err = mean_absolute_error(dict_cent[key],trocar[trocar_index])
			
			dict_abs_err[key] = abs_err

			trocar_index += 1

			continue

		else:

			for i in range(len(vect_cent)):

				if i in list_match_idx:

					continue

				cent_temp = np.array(vect_cent[i])

				abs_err = mean_absolute_error(cent_temp,trocar[trocar_index])

				if abs_err < match_max:

					match_max = abs_err
					match_index = i
						
			dict_cluster[key]  = vect_clustered[match_index]
			
			dict_cent[key] = vect_cent[match_index]

			dict_abs_err[key] = match_max

			list_match_idx.append(match_index)

		trocar_index += 1

		diff = len(vect_cent) - trocar_index
		diff1 = trocar.shape[0] - trocar_index

	y_true = np.arange(N_lines)
	y_true = y_true.tolist()

	y_pred = np.arange(N_lines)
	y_pred = y_pred.tolist()

	for key,value in dict_gt.items():

		for val in value:

			y_true[val] = key

	for key,value in dict_cluster.items():

		for val in value:

			y_pred[val] = key

	return y_pred,y_true,dict_cent,dict_abs_err

def estimate_trocar(vect_end,vect_start,lines_idx):


	vect_start_clustered = np.zeros((len(lines_idx),3),dtype=np.float32)
	vect_end_clustered = np.zeros((len(lines_idx),3),dtype=np.float32)	

	vect_start_clustered = vect_start[lines_idx]
	vect_end_clustered = vect_end[lines_idx]

	vect_rand_clustered = (vect_end_clustered - vect_start_clustered)/(SCALE_COEF1*(SCALE_COEF2+1))

	a,b = generate_coef(vect_rand_clustered, vect_end_clustered)

	final_sol,residuals_err = linear_least_squares(a,b,residuals=True)

	# # use np.linalg.lstsq
	# final_sol,_,_,_ = np.linalg.lstsq(a,b,rcond=None)
	# # print(final_sol)
	# final_sol = np.reshape(final_sol,(1,3))[0]
	# # print(final_sol)
	# residuals_err = np.linalg.norm(np.dot(a, final_sol) - b)

	# # use OLS
	# statsmodel_model = sm.OLS(b, a)
	# regression_results = statsmodel_model.fit()
	# final_sol = regression_results.params
	# residuals_err = regression_results.resid
	
	return final_sol, residuals_err, a, b

def read_ply(path):
	
	###Read Ply file and return it as numpy array
	plydata = PlyData.read(path)
	vertex_data = plydata['vertex'].data # numpy array with fields ['x', 'y', 'z']
	pts = np.zeros([vertex_data.size, 3])
	pts[:, 0] = vertex_data['x']
	pts[:, 1] = vertex_data['y']
	pts[:, 2] = vertex_data['z']

	return pts

def save_dataset(trocar, percentage,choice,sigma=5,upper_bound=150,N_lines=1000):

	lst = ['incorrect_data','noise','observed lines']
	data_path = '/home/bao/Downloads/Git/master_thesis/data'
	num_trocar = trocar.shape[0]

	if choice == lst[0]:
		list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
		pref = 'inc'
	elif choice == lst[1]:
		list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
		pref = 'sigma'
	else:
		list_noise_percentage = np.arange(start_range+step, end_range + step,step,dtype=np.uint8)
		list_noise_percentage = [element * 20 for element in list_noise_percentage]
		pref = 'lines'

	path = os.path.join(data_path,pref)

	for num in list_noise_percentage:

		if num != 0:

			break

		if choice == lst[0]:
			percentage[-1] = num/100
		elif choice ==lst[1]:
			sigma = num
		else:
			N_lines = num

		vect_start, vect_end, dict_gt = generate_data(N_lines=N_lines, percentage = percentage, trocar=trocar, scale1 = SCALE_COEF1, scale2 = SCALE_COEF2, sigma = sigma, upper_bound = upper_bound)

		file_name = pref + '_{:05d}'.format(num)

		np.savez(os.path.join(path,file_name+'.npz'), vect_start=vect_start, vect_end=vect_end)

		with open(os.path.join(path,file_name+'.pkl'), 'wb') as fp:
			pickle.dump(dict_gt, fp)

def load_dataset(file_name,gt_name):

	# lst_data = glob.glob('/home/bao/Documents/Git/master_thesis/data/inc/'+"*.npz")
	# lst_data.sort()

	# lst_gt = glob.glob('/home/bao/Documents/Git/master_thesis/data/inc/'+"*.pkl")
	# lst_gt.sort()

	with open(gt_name,'rb') as fp:
		dict_gt = pickle.load(fp)

	data = np.load(file_name)
	vect_start = data['vect_start']
	vect_end = data['vect_end']

	return vect_start,vect_end,dict_gt


def runMercuri():

	file_path = '/home/bao/Documents/Git/master_thesis/Radius.mat'
	result_path = '/home/bao/Documents/Git/master_thesis/result/mercuri'
	file_name = 'result.npz'

	lst_res = glob.glob(result_path + '/' + "*.npz")
	lst_res.sort()
	
	arr = np.empty((0,3), dtype=np.float32)

	u, pts = dataFromMercuri(file_path)

	data_len = int(u.shape[0]/5)

	# for i in range(data_len):

	# a,b = generate_coef(u[5*i:5*i+5,:], pts[5*i:5*i+5,:])
	a,b = generate_coef(u[0:5,:], pts[0:5,:])

	final_sol = linear_least_squares(a,b,residuals=False)

	final_sol = final_sol.reshape((1,3))

	arr = np.append(arr,final_sol,axis=0)

	print(final_sol)

	vect_start = pts - SCALE_COEF1*SCALE_COEF2*u
	vect_end = pts + SCALE_COEF1*SCALE_COEF2*u
	line_idx = {}
	line_idx["Trocar"] = [0,1,2,3,4]
	resid = lineseg_dist(final_sol[0],vect_start,vect_end)
	print(resid)	
	pts = read_ply("liver_views12345_ct0_wrp0_arap0_alterscheme3.ply")
	M = np.diag((1,-1,-1))
	pts = np.matmul(pts,M)
	visualize_model(pts= pts,vect_end = vect_end, vect_start = vect_start, line_idx = line_idx)

	# if lst_res:	

	# 	data = np.load(lst_res[0])

	# 	mercuri = data['mercuri']

	# 	mercuri = np.append(mercuri, arr, axis=0)

	# 	np.savez(lst_res[0], mercuri=mercuri)

	# else:

	# 	np.savez(os.path.join(result_path,file_name), mercuri=arr)

	# return



def draw_graph(choice,trocar):

	lst = ['incorrect_data','noise','observed lines']

	data_path = '/home/bao/Documents/Git/master_thesis/data'


	if choice == lst[0]:
		list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)

		pref_err = 'inc_err'

	elif choice == lst[1]:
		list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)

		pref_err = 'sigma_err'

	else:
		list_noise_percentage = np.arange(start_range+step, end_range + step,step,dtype=np.uint8)

		list_noise_percentage = [element * 20 for element in list_noise_percentage]

		pref_err = 'lines_err'

	path_err = os.path.join(data_path,pref_err)

	lst_err = glob.glob(path_err + '/' + "*.npz")
	lst_err.sort()

	list_abs_err = np.zeros((len(list_noise_percentage),trocar.shape[0]),dtype=np.float32)
	# list_acc = np.zeros((len(list_noise_percentage),num_trocar),dtype=np.float32)
	list_acc = np.zeros((len(list_noise_percentage),1),dtype=np.float32)
	list_trocar = np.zeros((len(list_noise_percentage),1),dtype=np.float32)
	list_mean = np.zeros((len(list_noise_percentage),1),dtype=np.float32)

	ite = 0

	data = np.load(lst_err[0])
	trocar1 = data['trocar1']
	trocar1[trocar1<0] = np.nan
	trocar2 = data['trocar2']
	trocar2[trocar2<0] = np.nan
	trocar3 = data['trocar3']
	trocar3[trocar3<0] = np.nan
	trocar4 = data['trocar4']
	trocar4[trocar4<0] = np.nan

	mean_err = data['mean_err']
	acc = data['acc']
	nt = data['num_trocar']


	length = int(trocar1.shape[0]/len(list_noise_percentage))


	for num in list_noise_percentage:
		
		temp1 = np.array([])
		temp2 = np.array([])
		temp3 = np.array([])
		temp4 = np.array([])

		for j in range(length):

			temp1 = np.append(temp1,trocar1[ite + len(list_noise_percentage)*j])
			temp2 = np.append(temp2,trocar2[ite + len(list_noise_percentage)*j])
			temp3 = np.append(temp3,trocar3[ite + len(list_noise_percentage)*j])
			temp4 = np.append(temp4,trocar4[ite + len(list_noise_percentage)*j])

			list_acc[ite] += acc[ite + len(list_noise_percentage)*j]
			list_trocar[ite] += nt[ite + len(list_noise_percentage)*j]
			list_mean[ite] += mean_err[ite+len(list_noise_percentage)*j]

		list_abs_err[ite,0] = np.nanmean(temp1)
		list_abs_err[ite,1] = np.nanmean(temp2)
		list_abs_err[ite,2] = np.nanmean(temp3)
		list_abs_err[ite,3] = np.nanmean(temp4)



		ite += 1	

	list_abs_err = list_abs_err/length
	list_acc = list_acc/length
	list_trocar = list_trocar/length
	list_mean = list_mean/length

	print(list_mean)
	#plot the result

	if choice == lst[1]:

		plt.figure(100),
		fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		axs.plot(list_noise_percentage, list_acc, 'o-')
		axs.xaxis.grid(True, which='major')
		axs.yaxis.grid(True, which='major')
		axs.set(xlabel='Amount of noise (mm)', ylabel='Clustering accuracy (%)')

		# fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		# axs.plot(list_noise_percentage, list_acc[:,0], 'r-')
		# axs.plot(list_noise_percentage, list_acc[:,1], 'b-')
		# axs.plot(list_noise_percentage, list_acc[:,2], 'g-')
		# axs.plot(list_noise_percentage, list_acc[:,3], 'm-')
		# axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
		# axs.set(xlabel='Amount of noise (mm)', ylabel='Clustering accuracy (%)')


		plt.figure(200),
		fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		axs.plot(list_noise_percentage, list_trocar, 'o-')
		axs.xaxis.grid(True, which='major')
		axs.yaxis.grid(True, which='major')
		axs.set(xlabel='Amount of noise (mm)', ylabel='Number of trocar predicted')

		plt.figure(300),
		fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		axs.plot(list_noise_percentage, list_abs_err[:,0], 'o-')
		axs.plot(list_noise_percentage, list_abs_err[:,1], 'o-')
		axs.plot(list_noise_percentage, list_abs_err[:,2], 'o-')
		axs.plot(list_noise_percentage, list_abs_err[:,3], 'o-')
		# axs.plot(list_noise_percentage, list_mean, 'o-')
		axs.plot(list_noise_percentage, np.nanmean(list_abs_err,axis=1), 'o-')

		axs.xaxis.grid(True, which='major')
		axs.yaxis.grid(True, which='major')
		axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4','Average'])
		axs.set(xlabel='Amount of noise (mm)', ylabel='Trocar position error (mm)')

	elif choice == lst[0]:

		plt.figure(100),
		fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		axs.plot(list_noise_percentage, list_acc, 'o-')
		axs.xaxis.grid(True, which='major')
		axs.yaxis.grid(True, which='major')
		axs.set(xlabel='Incorrect data (%)', ylabel='Clustering accuracy (%)')

		# fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		# axs.plot(list_noise_percentage, list_acc[:,0], 'r-')
		# axs.plot(list_noise_percentage, list_acc[:,1], 'b-')
		# axs.plot(list_noise_percentage, list_acc[:,2], 'g-')
		# axs.plot(list_noise_percentage, list_acc[:,3], 'm-')
		# axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
		# axs.set(xlabel='Incorrect data (%)', ylabel='Clustering accuracy (%)')

		plt.figure(200),
		fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		axs.plot(list_noise_percentage, list_trocar, 'o-')
		axs.xaxis.grid(True, which='major')
		axs.yaxis.grid(True, which='major')
		axs.set(xlabel='Incorrect data (%)', ylabel='Number of trocar predicted')

		plt.figure(300),
		fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		axs.plot(list_noise_percentage, list_abs_err[:,0], 'o-')
		axs.plot(list_noise_percentage, list_abs_err[:,1], 'o-')
		axs.plot(list_noise_percentage, list_abs_err[:,2], 'o-')
		axs.plot(list_noise_percentage, list_abs_err[:,3], 'o-')
		# axs.plot(list_noise_percentage, list_mean, 'o-')
		axs.plot(list_noise_percentage, np.nanmean(list_abs_err,axis=1), 'o-')
		axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4','Average'])
		axs.xaxis.grid(True, which='major')
		axs.yaxis.grid(True, which='major')
		axs.set(xlabel='Incorrect data (%)', ylabel='Trocar position error (mm)')

	else:
		plt.figure(100),
		fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		axs.plot(list_noise_percentage, list_acc, 'o-')
		axs.xaxis.grid(True, which='major')
		axs.yaxis.grid(True, which='major')
		axs.set(xlabel='Number of observed tool 3D axes', ylabel='Clustering accuracy (%)')

		# fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		# axs.plot(list_noise_percentage, list_acc[:,0], 'r-')
		# axs.plot(list_noise_percentage, list_acc[:,1], 'b-')
		# axs.plot(list_noise_percentage, list_acc[:,2], 'g-')
		# axs.plot(list_noise_percentage, list_acc[:,3], 'm-')
		# axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
		# axs.set(xlabel='Incorrect data (%)', ylabel='Clustering accuracy (%)')

		plt.figure(200),
		fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		axs.plot(list_noise_percentage, list_trocar, 'o-')
		axs.xaxis.grid(True, which='major')
		axs.yaxis.grid(True, which='major')
		axs.set(xlabel='Number of observed tool 3D axes', ylabel='Number of trocar predicted')

		plt.figure(300),
		fig, axs = plt.subplots(1, 1, figsize = (10, 4))
		axs.plot(list_noise_percentage, list_abs_err[:,0], 'o-')
		axs.plot(list_noise_percentage, list_abs_err[:,1], 'o-')
		axs.plot(list_noise_percentage, list_abs_err[:,2], 'o-')
		axs.plot(list_noise_percentage, list_abs_err[:,3], 'o-')
		# axs.plot(list_noise_percentage, list_mean, 'o-')
		axs.plot(list_noise_percentage, np.nanmean(list_abs_err,axis=1), 'o-')

		axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4','Average'])
		# axs.grid(True,linestyle='--')
		axs.xaxis.grid(True, which='major')
		axs.yaxis.grid(True, which='major')
		axs.set(xlabel='Number of observed tool 3D axes', ylabel='Trocar position error (mm)')
		
	plt.show()	


def test_case1(trocar, percentage,N_lines = 1000, sigma=5, upper_bound=150):

	# lst = ['incorrect_data','noise','observed lines']
	data_path = '/home/bao/Documents/Git/master_thesis/'
	num_trocar = trocar.shape[0]

	# if choice == lst[0]:
	# list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
	pref = 'data/sigma'
	pref_err = 'navid'
	# elif choice == lst[1]:
	# 	list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
	# 	pref = 'sigma'
	# 	pref_err = 'sigma_err'

	# else:
	# 	list_noise_percentage = np.arange(start_range+step, end_range + step,step,dtype=np.uint8)
	# 	list_noise_percentage = [element * 20 for element in list_noise_percentage]
	# 	pref = 'lines'
	# 	pref_err = 'lines_err'

	path = os.path.join(data_path,pref)

	lst_data = path + '/' + "sigma_00050.npz"

	lst_gt = path + '/' + "sigma_00050.pkl"

	path_err = os.path.join(data_path,pref_err)

	lst_err = glob.glob(data_path + '/' + "*.npz")
	lst_err.sort()

	list_abs_err = np.zeros((1,num_trocar),dtype=np.float32)
	# list_acc = np.zeros((len(list_noise_percentage),num_trocar),dtype=np.float32)
	# list_acc = np.zeros((1,1),dtype=np.float32)
	# list_trocar = np.zeros((1,1),dtype=np.uint8)


	# vect_start, vect_end, dict_gt = generate_data(N_lines=N_lines, percentage = percentage, trocar=trocar, scale1 = SCALE_COEF1, scale2 = SCALE_COEF2, sigma = sigma, upper_bound = upper_bound)
	# print(lst_data[ite])
	# print(lst_gt[ite])
	vect_start,vect_end,dict_gt = load_dataset(lst_data,lst_gt)

	# print(dict_gt)
	abs_err, acc_clustering, pred_trocar = ransac_new(trocar, vect_start, vect_end, dict_gt, N_lines = N_lines)

	# for i in range(len(acc_clustering)):

	# 	list_acc[ite,i] = acc_clustering[i]*100

	list_acc = acc_clustering*100
	list_trocar = pred_trocar

	for i in range(num_trocar):

		list_abs_err[0,i] = abs_err[i]

	# acc_clustering,num_trocar = ransac_new(trocar,percentage,N_lines,sigma,upper_bound,test_num_clus=True)
	# acc_clustering,num_trocar = ransac_new(trocar, vect_start, vect_end, dict_gt, N_lines = N_lines, test_num_clus=True)

	# list_trocar[ite] = num_trocar
	# list_acc[ite] = acc_clustering*100


	list_pos = np.copy(list_abs_err)
	list_pos[list_pos < 0 ] = np.nan
	
	if lst_err:
		
		data = np.load(lst_err[0])
		trocar1 = data['trocar1']
		trocar2 = data['trocar2']
		trocar3 = data['trocar3']
		trocar4 = data['trocar4']
		mean_err = data['mean_err']
		acc = data['acc']
		nt = data['num_trocar']
		trocar1	= np.append(trocar1,list_abs_err[0,0])
		trocar2	= np.append(trocar2,list_abs_err[0,1])
		trocar3	= np.append(trocar3,list_abs_err[0,2])
		trocar4	= np.append(trocar4,list_abs_err[0,3])
		mean_err = np.append(mean_err,np.nanmean(list_pos,axis=1))
		acc	= np.append(acc,list_acc)
		nt	= np.append(nt,list_trocar)

		np.savez(path_err+'.npz', trocar1=trocar1, trocar2=trocar2, trocar3=trocar3, trocar4=trocar4, mean_err=mean_err, acc=acc,num_trocar=nt)

	else:

		np.savez(path_err+'.npz', trocar1=list_abs_err[0,0], trocar2=list_abs_err[0,1], trocar3=list_abs_err[0,2], trocar4=list_abs_err[0,3], mean_err=np.nanmean(list_pos,axis=1), acc=list_acc,num_trocar=list_trocar)


###################################################################

if __name__ == '__main__':

	trocar = np.array([[-60,70,120],[-65,-70,120],[65,-75,120],[60,75,120]])

	# trocar = np.array([[30,68,125],[150,70,130],[35, 200,120]])
	# run_algo6(trocar,percentage)

	# percentage = np.array([0.35,0.27,0.18,0.2])
	# run_algo4(trocar,percentage)
	# run_algo5(trocar,percentage)
	percentage = np.array([0.4,0.3,0.2,0.1,0.1])
	# test_case(trocar, percentage, N_lines = 1000, sigma=15, upper_bound=150,use_inc=False)
	# vect_start, vect_end, dict_gt = generate_data(N_lines, percentage, trocar, scale1 = SCALE_COEF1, scale2 = SCALE_COEF2, sigma = 5, upper_bound = 150)
	
	# list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
	
	# estim_pt = np.array([-58.94028, 73.22261, 118.905426])

	# trocar_pt = np.array([[-60.1072, 70.414635,120.166214]])

	# min_list_idx_temp = lineseg_dist(estim_pt, vect_start, vect_end, list_idx_lines = list_idx, threshold = 8)

	# new_min_list_idx_temp = lineseg_dist(trocar_pt, vect_start, vect_end, list_idx_lines = list_idx, threshold = 8)

	# print(len(min_list_idx_temp))

	# print(len(new_min_list_idx_temp))


	# ransac_new(trocar = trocar, N_lines = 1000, vect_start = vect_start, vect_end = vect_end, dict_gt = dict_gt, test_num_clus=False)
	# ransac_new(trocar,percentage)
	choice = ['incorrect_data','noise','observed lines']

	draw_graph(choice[1],trocar)
	# for i in range(10000):
	# 	print(i)
		# test_case(trocar, percentage, N_lines = 1000, sigma=52, upper_bound=150, choice=choice[0])


	# save_dataset(trocar,percentage,choice[1])
	# load_dataset()

	# runMercuri()
	# for i in range(1000):
	# 	print(i)
	# 	test_case1(trocar, percentage, N_lines = 1000, sigma=5, upper_bound=150)
	


