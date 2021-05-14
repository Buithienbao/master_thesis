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

    coef1 = (np.dot(p1-p3,p4-p3)*np.dot(p4-p3,p2-p1) - np.dot(p1-p3,p2-p1)*np.dot(p4-p3,p4-p3))/(np.dot(p2-p1,p2-p1)*np.dot(p4-p3,p4-p3) - np.dot(p4-p3,p2-p1)*np.dot(p4-p3,p2-p1))
    coef2 = (np.dot(p1-p3,p4-p3) + coef1*np.dot(p4-p3,p2-p1))/np.dot(p4-p3,p4-p3)

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

def test_case(trocar, percentage, N_lines = 1000, sigma=5, upper_bound=150, use_inc=False):


	num_trocar = trocar.shape[0]

	if use_inc:
		list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
	else:
		list_noise_percentage = np.arange(start_range+step, end_range + step,step,dtype=np.uint8)
	
	list_abs_err = np.zeros((len(list_noise_percentage),num_trocar),dtype=np.float32)
	list_acc = np.zeros((len(list_noise_percentage),1),dtype=np.float32)
	list_trocar = np.zeros((len(list_noise_percentage),1),dtype=np.uint8)

	ite = 0

	for num in list_noise_percentage:

		if use_inc:
			percentage[-1] = num/100
		else:
			sigma = num
		
		vect_start, vect_end, dict_gt = generate_data(N_lines=N_lines, percentage = percentage, trocar=trocar, scale1 = SCALE_COEF1, scale2 = SCALE_COEF2, sigma = sigma, upper_bound = upper_bound)

		# abs_err, acc_clustering, num_trocar = ransac_new(trocar,percentage,N_lines,sigma,upper_bound)

		# list_acc[ite] = acc_clustering*100

		# list_trocar[ite] = num_trocar

		# for i in range(len(abs_err)):

		# 	list_abs_err[ite,i] = abs_err[i]

		# acc_clustering,num_trocar = ransac_new(trocar,percentage,N_lines,sigma,upper_bound,test_num_clus=True)
		acc_clustering,num_trocar = ransac_new(trocar, vect_start, vect_end, dict_gt, N_lines = N_lines, test_num_clus=True)

		list_trocar[ite] = num_trocar
		list_acc[ite] = acc_clustering*100

		ite += 1

	# #plot the result
	# plt.figure(100),
	# fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	# fig.suptitle('Relative error comparison')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,0], 'r-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,0], 'b-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,0], 'g-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,3,0], 'm-')
	# axs[0,0].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# axs[0,0].set(xlabel='Incorrect data (%)', ylabel='Relative error for X (%)')
	# # axs[0,0].set_title('Trocar 1')
	# axs[0,1].plot(list_noise_percentage, list_rela_err[:,0,1], 'r-')
	# axs[0,1].plot(list_noise_percentage, list_rela_err[:,1,1], 'b-')
	# axs[0,1].plot(list_noise_percentage, list_rela_err[:,2,1], 'g-')
	# axs[0,1].plot(list_noise_percentage, list_rela_err[:,3,1], 'm-')
	# axs[0,1].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# axs[0,1].set(xlabel='Incorrect data (%)', ylabel='Relative error for Y (%)')

	# axs[1,0].plot(list_noise_percentage, list_rela_err[:,0,2], 'r-')
	# axs[1,0].plot(list_noise_percentage, list_rela_err[:,1,2], 'b-')
	# axs[1,0].plot(list_noise_percentage, list_rela_err[:,2,2], 'g-')
	# axs[1,0].plot(list_noise_percentage, list_rela_err[:,3,2], 'm-')
	# axs[1,0].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# axs[1,0].set(xlabel='Incorrect data (%)', ylabel='Relative error for Z (%)')

	plt.figure(100),
	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	axs.plot(list_noise_percentage, list_acc, 'r-')
	axs.set(xlabel='Amount of noise (mm)', ylabel='Clustering accuracy (%)')

	plt.figure(200),
	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	axs.plot(list_noise_percentage, list_trocar, 'g-')
	axs.set(xlabel='Amount of noise (mm)', ylabel='Number of cluster predicted')

	# if use_inc:
	# 	plt.figure(300),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	# fig.suptitle('RMSE comparison')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,0], 'r-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,1], 'b-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,2], 'g-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,3], 'm-')
	# 	axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# 	axs.set(xlabel='Incorrect data (%)', ylabel='Trocar position error (mm)')

	# 	plt.figure(400),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	# fig.suptitle('Standard error comparison')
	# 	axs.plot(list_noise_percentage, list_acc, 'r-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,1], 'b-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,2], 'g-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,3], 'm-')
	# 	# axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# 	axs.set(xlabel='Incorrect data (%)', ylabel='Clustering accuracy (%)')
	# else:
	# 	plt.figure(300),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	# fig.suptitle('RMSE comparison')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,0], 'r-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,1], 'b-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,2], 'g-')
	# 	axs.plot(list_noise_percentage, list_abs_err[:,3], 'm-')
	# 	axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# 	axs.set(xlabel='Amount of noise (mm)', ylabel='Trocar position error (mm)')

	# 	plt.figure(400),
	# 	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# 	# fig.suptitle('Standard error comparison')
	# 	axs.plot(list_noise_percentage, list_acc, 'r-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,1], 'b-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,2], 'g-')
	# 	# axs.plot(list_noise_percentage, list_acc[:,3], 'm-')
	# 	# axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# 	axs.set(xlabel='Amount of noise (mm)', ylabel='Clustering accuracy (%)')

	plt.show()	



def ransac_new(trocar, vect_start, vect_end, dict_gt, N_lines = 1000, test_num_clus=False):

	num_trocar = trocar.shape[0]

	dict_cluster = {}

	list_abs_err = np.zeros((num_trocar,1),dtype=np.float32)

	num_trials = 100000000
	sample_count = 0
	sample_size = 2
	P_min = 0.99
	temp_per = 0

	list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
	# remove_idx = []
	vect_clustered = []
	threshold_dist = 15
	threshold_inliers = 60
	vect_cent = []

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

			count_cluster+=1
			print("Cluster " + str(count_cluster) + " found.")
			vect_cent.append(center_point_temp)
			vect_clustered.append(min_list_idx_temp.tolist())
			list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
			flat_list = [item for sublist in vect_clustered for item in sublist]
			flat_list = np.array(flat_list)
			# print(sorted(np.unique(flat_list)))
			list_idx = list_idx[~np.isin(list_idx,flat_list)]
			
			if not len(list_idx):

				print("Last cluster. Length: 0")
				break

			elif len(list_idx) < 3:
				print("Last cluster. Length: ",len(list_idx))
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
		print("Last cluster. Length: ",len(list_idx))
		vect_clustered.append(list_idx.tolist())
	

	# print("--- %s seconds ---" % (time.time() - start_time))


	y_true,y_pred,center_pts = flattenClusteringResult(dict_gt,dict_cluster,vect_clustered,vect_cent,N_lines)
	# plot_cfs_matrix(y_true,y_pred,list(dict_gt.keys()))
	acc_clustering = accuracy_score(y_true,y_pred)
	# print(acc_clustering)

	
	if not test_num_clus:

		ite = 0

		for key,value in dict_cluster.items():

			if ite == num_trocar:

				break

			final_sol = center_pts[key]

			abs_err = mean_absolute_error(final_sol,trocar[ite])
			
			list_abs_err[ite] = abs_err

			ite += 1


		# pts = read_ply("liver_simplified.ply")


		# visualize_model(pts=pts,trocar=trocar,vect_end=vect_end,vect_start=vect_start,line_idx=dict_gt,gt=True)
		
		# visualize_model(pts=pts,trocar=trocar,vect_end=vect_end,vect_start=vect_start,line_idx=dict_cluster,gt=False)
			# visualize_model(pts=pts)

		return list_abs_err, acc_clustering, len(vect_clustered)

	else:

		return acc_clustering,len(vect_clustered)

def flattenClusteringResult(dict_gt,dict_cluster,vect_clustered,vect_cent,N_lines):

	"""
	Convert from dictionary to 1D numpy array
	"""
	dict_cent = {}
	for key,value in dict_gt.items():

		gt_temp = set(value)
		match_max = 0
		match_index = 0

		for i in range(len(vect_clustered)):

			vect_temp = set(vect_clustered[i])

			match = len(gt_temp & vect_temp)

			if match > match_max:

				match_max = match
				match_index = i

		dict_cluster[key]  = vect_clustered[match_index]

		if match_index >= len(vect_cent):
			
			dict_cent[key] = 0

		else:

			dict_cent[key] = vect_cent[match_index]

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

	return y_pred,y_true,dict_cent

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

###################################################################

if __name__ == '__main__':

	trocar = np.array([[-60,70,120],[-65,-70,120],[65,-75,120],[60,75,120]])

	# trocar = np.array([[30,68,125],[150,70,130],[35, 200,120]])
	# run_algo6(trocar,percentage)

	# percentage = np.array([0.35,0.27,0.18,0.2])
	# run_algo4(trocar,percentage)
	# run_algo5(trocar,percentage)
	percentage = np.array([0.4,0.3,0.2,0.1,0.2])
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
	
	test_case(trocar, percentage, N_lines = 1000, sigma=10, upper_bound=150, use_inc=False)