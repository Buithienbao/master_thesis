import numpy as np 
import shapely
from generate_data import *
from scipy.optimize import least_squares,leastsq
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import math
from sklearn.utils import shuffle
from skimage.draw import ellipsoid
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from pyntcloud import PyntCloud
from plyfile import PlyData
from itertools import product, combinations, cycle
LOOP_NUM = 10
N_lines = 1000
# percentage = 0.2
# num_outliers = int(N_lines*percentage)

start_range = 0
end_range = 35
# step = (end_range - start_range)/10
step = 5
SCALE_COEF = 10
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

	return np.abs(pred-gt)/gt*100

def abs_err_calc(pred,gt):

	return np.abs(pred-gt)

def eudist_err_calc(pred,gt):

	return np.linalg.norm(pred-gt)


def run_algo():

	p0 = np.array([50,50,50]).astype(np.float32)

	a_train,b_train,gt,_ = generate_perfect_data(N_lines = N_lines)

	list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
	# print(list_noise)
	list_rela_err_L1 = []

	list_abs_err_L1 = []

	list_eu_err_L1 = []

	list_rela_err_L2 = []

	list_abs_err_L2 = []

	list_eu_err_L2 = []

	for num in list_noise_percentage:
		
		sum_res_soft_L1 = 0

		sum_res_soft_L2 = 0
		
		if num == 0:

			#Perfect data result

			for i in range(LOOP_NUM):

				res_soft_l1 = least_squares(lineseg_dist, p0, loss='soft_l1', f_scale=0.1,

				                            args=(a_train, b_train))

				res_soft_l2 = least_squares(lineseg_dist, p0, loss='linear', f_scale=0.1,

				                            args=(a_train, b_train))

				sum_res_soft_L1 += res_soft_l1.x

				sum_res_soft_L2 += res_soft_l2.x

			mean_res_soft_L1 = sum_res_soft_L1/LOOP_NUM

			mean_res_soft_L2 = sum_res_soft_L2/LOOP_NUM
			
			rela_err_L1 = relative_err_calc(mean_res_soft_L1,gt)
			rela_err_L2 = relative_err_calc(mean_res_soft_L2,gt)

			abs_err_L1 = abs_err_calc(mean_res_soft_L1,gt)
			abs_err_L2 = abs_err_calc(mean_res_soft_L2,gt)

			eu_err_L1 = eudist_err_calc(mean_res_soft_L1,gt)
			eu_err_L2 = eudist_err_calc(mean_res_soft_L2,gt)

			list_rela_err_L1.append(rela_err_L1)
			list_rela_err_L2.append(rela_err_L2)

			list_abs_err_L1.append(abs_err_L1)
			list_abs_err_L2.append(abs_err_L2)

			list_eu_err_L1.append(eu_err_L1)
			list_eu_err_L2.append(eu_err_L2)

		else:

			#Outlier data result

			# outlier_a, outlier_b,_ = generate_outliers(N_outliers = num)

			# c_train = np.concatenate((a_train,outlier_a))

			# d_train = np.concatenate((b_train,outlier_b))

			for i in range(LOOP_NUM):

				# a_train_with_noise = add_gaussian_noise(a_train,mean=0,var=num,percentage=0.2)
				a_train_with_noise = add_gaussian_noise(a_train,mean=0,var=20,percentage=num/100)

				res_soft_l1 = least_squares(lineseg_dist, p0, loss='soft_l1', f_scale=0.1,

				                            args=(a_train_with_noise, b_train))

				res_soft_l2 = least_squares(lineseg_dist, p0, loss='linear', f_scale=0.1,

				                            args=(a_train_with_noise, b_train))

				sum_res_soft_L1 += res_soft_l1.x

				sum_res_soft_L2 += res_soft_l2.x

			mean_res_soft_L1 = sum_res_soft_L1/LOOP_NUM

			mean_res_soft_L2 = sum_res_soft_L2/LOOP_NUM
			
			rela_err_L1 = relative_err_calc(mean_res_soft_L1,gt)
			rela_err_L2 = relative_err_calc(mean_res_soft_L2,gt)

			abs_err_L1 = abs_err_calc(mean_res_soft_L1,gt)
			abs_err_L2 = abs_err_calc(mean_res_soft_L2,gt)

			eu_err_L1 = eudist_err_calc(mean_res_soft_L1,gt)
			eu_err_L2 = eudist_err_calc(mean_res_soft_L2,gt)

			list_rela_err_L1.append(rela_err_L1)
			list_rela_err_L2.append(rela_err_L2)

			list_abs_err_L1.append(abs_err_L1)
			list_abs_err_L2.append(abs_err_L2)

			list_eu_err_L1.append(eu_err_L1)
			list_eu_err_L2.append(eu_err_L2)

	# print('list_rela_err_L1: ',list_rela_err_L1)
	# print('list_rela_err_L2: ',list_rela_err_L2)

	# print('list_abs_err_L1: ',list_abs_err_L1)
	# print('list_abs_err_L2: ',list_abs_err_L2)

	# print('list_eu_err_L1: ',list_eu_err_L1)
	# print('list_eu_err_L2: ',list_eu_err_L2)

	list_rela_err_L1 = np.array(list_rela_err_L1)
	list_rela_err_L2 = np.array(list_rela_err_L2)
	list_abs_err_L1 = np.array(list_abs_err_L1)
	list_abs_err_L2 = np.array(list_abs_err_L2)

	#plot the result
	fig, axs = plt.subplots(3, 2, figsize = (10, 4))

	axs[0,0].plot(list_noise_percentage, list_rela_err_L1[:,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err_L1[:,1], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err_L1[:,2], 'g-')
	axs[0,0].legend(['L1-norm -- Relative error for X','L1-norm -- Relative error for Y','L1-norm -- Relative error for Z'])
	axs[0,0].set(xlabel='Outliers (%)', ylabel='Relative error (%)')

	axs[0,1].plot(list_noise_percentage, list_rela_err_L2[:,0], 'r-')
	axs[0,1].plot(list_noise_percentage, list_rela_err_L2[:,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_rela_err_L2[:,2], 'g-')
	axs[0,1].legend(['L2-norm -- Relative error for X','L2-norm -- Relative error for Y','L2-norm -- Relative error for Z'])
	axs[0,1].set(xlabel='Outliers (%)', ylabel='Relative error (%)')

	axs[1,0].plot(list_noise_percentage, list_abs_err_L1[:,0], 'r-')
	axs[1,0].plot(list_noise_percentage, list_abs_err_L1[:,1], 'b-')
	axs[1,0].plot(list_noise_percentage, list_abs_err_L1[:,2], 'g-')
	axs[1,0].legend(['L1-norm -- Absolute error for X','L1-norm -- Absolute error for Y','L1-norm -- Absolute error for Z'])
	axs[1,0].set(xlabel='Outliers (%)', ylabel='Absolute error')

	axs[1,1].plot(list_noise_percentage, list_abs_err_L2[:,0], 'r-')
	axs[1,1].plot(list_noise_percentage, list_abs_err_L2[:,1], 'b-')
	axs[1,1].plot(list_noise_percentage, list_abs_err_L2[:,2], 'g-')
	axs[1,1].legend(['L2-norm -- Absolute error for X','L2-norm -- Absolute error for Y','L2-norm -- Absolute error for Z'])
	axs[1,1].set(xlabel='Outliers (%)', ylabel='Absolute error')


	axs[2,0].plot(list_noise_percentage, list_eu_err_L1, 'r--')
	axs[2,0].plot(list_noise_percentage, list_eu_err_L2, 'b-')
	axs[2,0].legend(['L1-norm -- RMSE', 'L2-norm -- RMSE'])
	axs[2,0].set(xlabel='Outliers (%)', ylabel='RMSE')

	plt.show()

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

def run_algo1():

	final_sol = 0
		
	unit_vect = []

	for i in range(N_lines):

		unit_vect_temp = random_unit_vector()
		unit_vect.append(unit_vect_temp)

	unit_vect = np.asarray(unit_vect)

	gt = get_trocar_gt()

	point = random_point_based_on_unit_vect(unit_vect, gt)

	a,b = generate_coef(unit_vect,point)
	
	list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)

	list_rela_err = []

	list_abs_err = []

	list_eu_err = []

	for num in list_noise_percentage:

		if num == 0:

			final_sol = 0
				
			for i in range(LOOP_NUM):

				x = linear_least_squares(a,b)
				final_sol += x
			
			final_sol = final_sol/LOOP_NUM

			rela_err = relative_err_calc(final_sol,gt)

			abs_err = abs_err_calc(final_sol,gt)

			eu_err = eudist_err_calc(final_sol,gt)

			list_rela_err.append(rela_err)

			list_abs_err.append(abs_err)

			list_eu_err.append(eu_err)

		else:

			final_sol = 0

			point_with_noise = add_gaussian_noise(point,mean=0,var=20,percentage=num/100)

			_,b_with_noise = generate_coef(unit_vect,point_with_noise)

			for i in range(LOOP_NUM):

				x = linear_least_squares(a,b_with_noise)
				final_sol += x
			
			final_sol = final_sol/LOOP_NUM

			rela_err = relative_err_calc(final_sol,gt)

			abs_err = abs_err_calc(final_sol,gt)

			eu_err = eudist_err_calc(final_sol,gt)

			list_rela_err.append(rela_err)

			list_abs_err.append(abs_err)

			list_eu_err.append(eu_err)


	list_rela_err = np.asarray(list_rela_err)

	list_abs_err = np.asarray(list_abs_err)

	#plot the result
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))

	axs[0,0].plot(list_noise_percentage, list_rela_err[:,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,1], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,2], 'g-')
	axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	axs[0,0].set(xlabel='Outliers (%)', ylabel='Relative error (%)')

	axs[1,0].plot(list_noise_percentage, list_abs_err[:,0], 'r-')
	axs[1,0].plot(list_noise_percentage, list_abs_err[:,1], 'b-')
	axs[1,0].plot(list_noise_percentage, list_abs_err[:,2], 'g-')
	axs[1,0].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	axs[1,0].set(xlabel='Outliers (%)', ylabel='Absolute error')

	axs[1,1].plot(list_noise_percentage, list_eu_err, 'r--')
	axs[1,1].legend(['RMSE'])
	axs[1,1].set(xlabel='Outliers (%)', ylabel='RMSE')

	plt.show()

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


def run_algo2(trocar,percentage):

	# Generate lines to each trocar

	vect_end = np.empty((0,3),dtype=np.float32)	
	vect_start = np.empty((0,3),dtype=np.float32)	
	N_lines = 1000

	num_trocar = trocar.shape[0]

	for i in range(num_trocar):

		end_temp, start_temp,_,_ = generate_perfect_data(int(N_lines*percentage[i]), trocar[i]) 
		vect_end = np.append(vect_end,end_temp,axis=0)
		vect_start = np.append(vect_start,start_temp,axis=0)

	# vect_end_with_noise = add_gaussian_noise(vect_end, percentage=percentage[-1])

	outlier_end, outlier_start,_ = generate_outliers(int(N_lines*percentage[-1]), trocar[0])
	vect_end =np.append(vect_end,outlier_end,axis=0)
	vect_start= np.append(vect_start,outlier_start,axis=0)

	
	# print(vect_end.shape)
	# print(vect_start.shape)
	vect_end,vect_start = shuffle(vect_end,vect_start)

	# Define Ransac params
	P_min = 0.999999999 
	sample_size = 2
	
	vect_clustered = [[] for i in range(num_trocar)]
	# vect_start_clustered = [[]]*num_trocar



	temp_per = 0
	list_idx = np.random.choice(N_lines, size=N_lines, replace=False)		

	for i in range(num_trocar):		

		P_outlier = 1 - percentage[i]/(1-temp_per)

		N_trial = int(math.log(1-P_min)/math.log(1-(1-P_outlier)**sample_size))
		count = 0
		N_data = int(N_lines*percentage[i])
		# print(N_data)
		while(count < N_data):

			temp_idx = []
			list_error = []

			for j in range(N_trial):

				# line_temp1, line_temp2 = zip(*random.sample(list(zip(vect_end, vect_start)), 2))
				idx1 = random.choice(list_idx)
				list_idx = np.delete(list_idx,np.where(list_idx==idx1))
				
				idx2 = random.choice(list_idx)
				list_idx = np.delete(list_idx,np.where(list_idx==idx2))

				temp_idx.append(idx1)
				temp_idx.append(idx2)

				estim_pt = find_intersection_3d_lines(vect_end[idx1],vect_start[idx1],vect_end[idx2],vect_start[idx2])

				error = np.linalg.norm(trocar[i]-estim_pt)

				list_error.append(error)

			idx_min = list_error.index(min(list_error))

			line1_idx = temp_idx[idx_min*2]
			line2_idx = temp_idx[idx_min*2+1]

			temp_vect_clustered = vect_clustered[i]
			temp_vect_clustered.append(line1_idx)
			temp_vect_clustered.append(line2_idx)

			temp_idx.pop(idx_min*2)
			temp_idx.pop(idx_min*2)

			list_idx = np.append(list_idx,np.asarray(temp_idx))
			list_idx = shuffle(list_idx)

			count += 2


		temp_per += percentage[i]

	# outlier_idx = list_idx

	# print(len(outlier_idx))
	# print(len(vect_clustered[0]))
	# print(len(vect_clustered[1]))
	# print(len(vect_clustered[2]))

	list_rela_err = []

	list_abs_err = []

	list_eu_err = []

	for i in range(num_trocar):

		vect_start_clustered = np.zeros((len(vect_clustered[i]),3),dtype=np.float32)
		vect_end_clustered = np.zeros((len(vect_clustered[i]),3),dtype=np.float32)
		
		for j in range(len(vect_clustered[i])):

			vect_start_clustered[j] = vect_start[vect_clustered[i][j]]
			vect_end_clustered[j] = vect_end[vect_clustered[i][j]]

		vect_rand_clustered = vect_end_clustered - vect_start_clustered

		a,b = generate_coef(vect_rand_clustered, vect_end_clustered)
		
		final_sol = 0
		residuals_err = 0
		for k in range(LOOP_NUM):

			x,residuals = linear_least_squares(a,b,residuals=True)
			final_sol += x
			residuals_err += residuals	

		final_sol = final_sol/LOOP_NUM

		rela_err = relative_err_calc(final_sol,trocar[i])

		abs_err = abs_err_calc(final_sol,trocar[i])

		eu_err = eudist_err_calc(final_sol,trocar[i])

		residuals_err = residuals_err/LOOP_NUM

		a_pinv = np.linalg.inv(np.dot(a.T,a))
		var_mtrx_x = np.dot(residuals_err,a_pinv)/(a.shape[0]-3+1)
		diagonal_x = np.diagonal(var_mtrx_x)
		std_err_x = np.linalg.norm(diagonal_x)
		
		temp_mtrx = np.dot(var_mtrx_x,a.T)

		var_mtrx_y = np.dot(a, temp_mtrx)
		diagonal_y = np.diagonal(var_mtrx_y)
		std_err_y = np.linalg.norm(diagonal_y)

		print("Trocar ground truth {}: {}".format(i,trocar[i]))
		print("Estimated trocar: ",final_sol)
		print("Relative error for X,Y,Z respectively (%): {} - {} - {}".format(rela_err[0],rela_err[1],rela_err[2]))
		print("Absolute error for X,Y,Z respectively: {} - {} - {}".format(abs_err[0],abs_err[1],abs_err[2]))
		print("Root mean square error: ",eu_err)
		print("Standard error x: ",std_err_x)
		print("Standard error y: ",std_err_y)

def run_algo3(trocar, percentage):

	# Generate lines to each trocar

	vect_end = np.empty((0,3),dtype=np.float32)	
	vect_start = np.empty((0,3),dtype=np.float32)	
	N_lines = 1000
	list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)

	num_trocar = trocar.shape[0]

	for i in range(num_trocar):

		end_temp, start_temp,_,_ = generate_perfect_data(int(N_lines*percentage[i]), trocar[i]) 
		vect_end = np.append(vect_end,end_temp,axis=0)
		vect_start = np.append(vect_start,start_temp,axis=0)

	# vect_end_with_noise = add_gaussian_noise(vect_end, percentage=percentage[-1])

	outlier_end, outlier_start,_ = generate_outliers(int(N_lines*percentage[-1]), trocar[0])
	
	list_rela_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_abs_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_eu_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)
	list_std_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)

	ite = 0

	for num in list_noise_percentage:

		outlier_end_noise = add_gaussian_noise(outlier_end,var=num,percentage=1)

		vect_end =np.append(vect_end,outlier_end_noise,axis=0)
		vect_start= np.append(vect_start,outlier_start,axis=0)

		
		# print(vect_end.shape)
		# print(vect_start.shape)
		vect_end,vect_start = shuffle(vect_end,vect_start)

		# Define Ransac params
		P_min = 0.999999
		sample_size = 2
		
		vect_clustered = [[] for i in range(num_trocar)]
		# vect_start_clustered = [[]]*num_trocar



		temp_per = 0
		list_idx = np.random.choice(N_lines, size=N_lines, replace=False)		
		remove_idx = []
		for i in range(num_trocar):		

			P_outlier = 1 - percentage[i]/(1-temp_per)

			N_trial = int(math.log(1-P_min)/math.log(1-(1-P_outlier)**sample_size))
			count = 0
			N_data = int(N_lines*percentage[i])
			# print(N_data)

			# while(count < N_data):

			dist = []
			min_list_idx = np.zeros((N_trial,N_data),dtype=np.uint8)

			for j in range(N_trial):

				# line_temp1, line_temp2 = zip(*random.sample(list(zip(vect_end, vect_start)), 2))
				idx1 = random.choice(list_idx)
				list_idx = np.delete(list_idx,np.where(list_idx==idx1))
				
				idx2 = random.choice(list_idx)
				list_idx = np.delete(list_idx,np.where(list_idx==idx2))

				estim_pt = find_intersection_3d_lines(vect_end[idx1],vect_start[idx1],vect_end[idx2],vect_start[idx2])

				dist_temp, min_list_idx_temp = lineseg_dist(estim_pt,vect_start,vect_end,index = N_data)

				min_list_idx[j] = min_list_idx_temp
				
				dist.append(dist_temp)


			idx_min = dist.index(min(dist))
			remove_idx.append(min_list_idx[idx_min])

			vect_clustered[i] = min_list_idx[idx_min]

			list_idx = np.arange(N_lines)
			flat_list = [item for sublist in remove_idx for item in sublist]
			list_idx = np.delete(list_idx, np.asarray(flat_list))
			list_idx = shuffle(list_idx)
			
			temp_per += percentage[i]

		for i in range(num_trocar):

			vect_start_clustered = np.zeros((len(vect_clustered[i]),3),dtype=np.float32)
			vect_end_clustered = np.zeros((len(vect_clustered[i]),3),dtype=np.float32)
			
			for j in range(len(vect_clustered[i])):

				vect_start_clustered[j] = vect_start[vect_clustered[i][j]]
				vect_end_clustered[j] = vect_end[vect_clustered[i][j]]

			vect_rand_clustered = vect_end_clustered - vect_start_clustered

			a,b = generate_coef(vect_rand_clustered, vect_end_clustered)
			
			final_sol = 0
			residuals_err = 0

			for k in range(LOOP_NUM):

				# x,residuals = linear_least_squares(a,b,residuals=True)
				x,residuals = np.linalg.lstsq(a,b)
				final_sol += x
				residuals_err += residuals	

			final_sol = final_sol/LOOP_NUM

			rela_err = relative_err_calc(final_sol,trocar[i])

			abs_err = abs_err_calc(final_sol,trocar[i])

			eu_err = eudist_err_calc(final_sol,trocar[i])


			covar = np.matrix(np.dot(a.T, a)).I
			residuals_err = residuals_err/LOOP_NUM
			# a_pinv = np.linalg.pinv(a.T)
			var_mtrx = np.dot(residuals_err,covar)/(a.shape[0]-3+1)
			diagonal = np.diagonal(var_mtrx)
			std_err = np.linalg.norm(diagonal)
			u,s,vh = np.linalg.svd(var_mtrx, full_matrices=True)
			# print("Singular values: ",s)
			print("Trocar ground truth {}: {}".format(i,trocar[i]))
			print("Estimated trocar: ",final_sol)
			print("Relative error for X,Y,Z respectively (%): {} - {} - {}".format(rela_err[0],rela_err[1],rela_err[2]))
			print("Absolute error for X,Y,Z respectively: {} - {} - {}".format(abs_err[0],abs_err[1],abs_err[2]))
			print("Root mean square error: ",eu_err)
			print("Covariance matrix associated to the estimated trocar: ",var_mtrx)
			print("Standard error: ",std_err)

			list_rela_err[ite,i,:] = rela_err
			list_abs_err[ite,i,:] = abs_err
			list_eu_err[ite,i] = eu_err
			list_std_err[ite,i] = std_err

		ite += 1
	#plot the result
	plt.figure(100),
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	fig.suptitle('Trocar 1')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,1], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,2], 'g-')
	axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	axs[0,0].set(xlabel='Noise variance', ylabel='Relative error (%)')
	# axs[0,0].set_title('Trocar 1')
	
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,0], 'r-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,2], 'g-')
	axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	axs[0,1].set(xlabel='Noise variance', ylabel='Absolute error (mm)')
	# axs[0,1].set_title('Trocar 1')

	axs[1,0].plot(list_noise_percentage, list_eu_err[:,0], 'r--')
	axs[1,0].legend(['RMSE'])
	axs[1,0].set(xlabel='Noise variance', ylabel='RMSE (mm)')
	# axs[0,2].set_title('Trocar 1')
	
	axs[1,1].plot(list_noise_percentage, list_std_err[:,0], 'r--')
	axs[1,1].legend(['Standard error'])
	axs[1,1].set(xlabel='Noise variance', ylabel='Standard error (mm)')
	# axs[0,3].set_title('Trocar 1')
	
	plt.figure(200),
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	fig.suptitle('Trocar 2')

	axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,1], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,2], 'g-')
	axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	axs[0,0].set(xlabel='Noise variance', ylabel='Relative error (%)')
	# axs[1,0].set_title('Trocar 2')

	axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,0], 'r-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,2], 'g-')
	axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	axs[0,1].set(xlabel='Noise variance', ylabel='Absolute error (mm)')
	# axs[1,1].set_title('Trocar 2')

	axs[1,0].plot(list_noise_percentage, list_eu_err[:,1], 'r--')
	axs[1,0].legend(['RMSE'])
	axs[1,0].set(xlabel='Noise variance', ylabel='RMSE (mm)')
	# axs[1,2].set_title('Trocar 2')

	axs[1,1].plot(list_noise_percentage, list_std_err[:,1], 'r--')
	axs[1,1].legend(['Standard error'])
	axs[1,1].set(xlabel='Noise variance', ylabel='Standard error (mm)')
	# axs[1,3].set_title('Trocar 2')
	
	plt.figure(300),
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	fig.suptitle('Trocar 3')

	axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,1], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,2], 'g-')
	axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	axs[0,0].set(xlabel='Noise variance', ylabel='Relative error (%)')
	# axs[2,0].set_title('Trocar 3')

	axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,0], 'r-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,2], 'g-')
	axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	axs[0,1].set(xlabel='Noise variance', ylabel='Absolute error (mm)')
	# axs[2,1].set_title('Trocar 3')

	axs[1,0].plot(list_noise_percentage, list_eu_err[:,2], 'r--')
	axs[1,0].legend(['RMSE'])
	axs[1,0].set(xlabel='Noise variance', ylabel='RMSE (mm)')	
	# axs[2,2].set_title('Trocar 3')

	axs[1,1].plot(list_noise_percentage, list_std_err[:,2], 'r--')
	axs[1,1].legend(['Standard error'])
	axs[1,1].set(xlabel='Noise variance', ylabel='Standard error (mm)')
	# axs[2,3].set_title('Trocar 3')

	plt.show()

def run_algo4(trocar, percentage):

	# Generate lines to each trocar

	vect_end = np.empty((0,3),dtype=np.float32)	
	vect_start = np.empty((0,3),dtype=np.float32)	
	N_lines = 1000
	list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)

	num_trocar = trocar.shape[0]

	for i in range(num_trocar):

		end_temp, start_temp,_,_ = generate_perfect_data(int(N_lines*percentage[i]), trocar[i]) 
		vect_end = np.append(vect_end,end_temp,axis=0)
		vect_start = np.append(vect_start,start_temp,axis=0)

	# vect_end_with_noise = add_gaussian_noise(vect_end, percentage=percentage[-1])

	outlier_end, outlier_start,_ = generate_outliers(int(N_lines*0.1), trocar[0])
	outlier_end1,outlier_start1,_ = generate_outliers(int(N_lines*0.06), trocar[1])
	outlier_end2,outlier_start2,_ = generate_outliers(int(N_lines*0.04), trocar[2])
	list_rela_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_abs_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_eu_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)
	list_std_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)

	ite = 0
	
	for num in list_noise_percentage:



		outlier_end_noise = add_gaussian_noise(outlier_end,var=num,percentage=1)
		outlier_end_noise1 = add_gaussian_noise(outlier_end1,var=num,percentage=1)
		outlier_end_noise2 = add_gaussian_noise(outlier_end2,var=num,percentage=1)

		vect_end =np.append(vect_end,outlier_end_noise,axis=0)
		vect_start= np.append(vect_start,outlier_start,axis=0)
		vect_end =np.append(vect_end,outlier_end_noise1,axis=0)
		vect_start= np.append(vect_start,outlier_start1,axis=0)
		vect_end =np.append(vect_end,outlier_end_noise2,axis=0)
		vect_start= np.append(vect_start,outlier_start2,axis=0)
		
		# print(vect_end.shape)
		# print(vect_start.shape)
		vect_end,vect_start = shuffle(vect_end,vect_start)

		# Define Ransac params
		P_min = 0.999999
		sample_size = 2
		
		vect_clustered = [[] for i in range(num_trocar)]
		# vect_start_clustered = [[]]*num_trocar



		temp_per = 0
		list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
		remove_idx = []
		for i in range(num_trocar):		

			list_idx_copy = list_idx.copy()

			P_outlier = 1 - percentage[i]/(1-temp_per)

			N_trial = int(math.log(1-P_min)/math.log(1-(1-P_outlier)**sample_size))
			count = 0
			N_data = int(N_lines*percentage[i])
			# print(N_data)

			# while(count < N_data):

			min_list_idx = np.array([], dtype=np.uint8).reshape(0,N_data)
			threshold = 10
			temp_idx = np.zeros(N_trial,dtype=np.uint8)
			###
			# while (threshold > 0.01):
			
			dist = []
			# print(N_trial)
			for j in range(N_trial):

				idx1 = random.choice(list_idx)
				list_idx = np.delete(list_idx,np.where(list_idx==idx1))
				
				idx2 = random.choice(list_idx)
				list_idx = np.delete(list_idx,np.where(list_idx==idx2))

				temp_idx = np.append(temp_idx,idx1)
				temp_idx = np.append(temp_idx,idx2)

				estim_pt = find_intersection_3d_lines(vect_end[idx1],vect_start[idx1],vect_end[idx2],vect_start[idx2])

				dist_temp, min_list_idx_temp = lineseg_dist(estim_pt,vect_start,vect_end,index = N_data, list_idx_lines = list_idx_copy)

				# print(len(np.unique(min_list_idx_temp)))
				min_list_idx = np.vstack([min_list_idx,min_list_idx_temp])
				
				dist.append(dist_temp)

			idx_min = dist.index(min(dist))

			threshold = dist[idx_min]

			# list_idx = np.append(list_idx,temp_idx)
			
			# list_idx = shuffle(list_idx)
			###
			# print(dist)
			# print(idx_min)
			# print(min_list_idx)
			remove_idx.append(min_list_idx[idx_min])

			vect_clustered[i] = np.copy(min_list_idx[idx_min])
			# print(np.sort(np.unique(vect_clustered[i])))
			print(len(np.unique(vect_clustered[i])))
			list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
			flat_list = [item for sublist in remove_idx for item in sublist]
			flat_list = np.array(flat_list)
			# print(sorted(np.unique(flat_list)))
			print(len(np.unique(flat_list)))
			list_idx = list_idx[~np.isin(list_idx,flat_list)]
			# list_idx = shuffle(list_idx)
			temp_per += percentage[i]
			# print(sorted(np.unique(list_idx)))
			print(len(np.unique(list_idx)))
			# print(np.sort(list_idx))
			# print(list_idx)
			# print(len(list_idx))
			# print(len(flat_list))
			# print(len(np.unique(min_list_idx[idx_min])))
			# print(list_idx)
		# for i in range(num_trocar):
		# 	print(vect_clustered[i])

		for i in range(num_trocar):

			vect_start_clustered = np.zeros((len(vect_clustered[i]),3),dtype=np.float32)
			vect_end_clustered = np.zeros((len(vect_clustered[i]),3),dtype=np.float32)
			
			for j in range(len(vect_clustered[i])):

				vect_start_clustered[j] = vect_start[vect_clustered[i][j]]
				vect_end_clustered[j] = vect_end[vect_clustered[i][j]]

			vect_rand_clustered = vect_end_clustered - vect_start_clustered

			a,b = generate_coef(vect_rand_clustered, vect_end_clustered)

			final_sol = 0
			residuals_err = 0

			for k in range(LOOP_NUM):

				x,residuals = linear_least_squares(a,b,residuals=True)
				# x,residuals = np.linalg.lstsq(a,b,rcond=None)
				final_sol += x
				residuals_err += residuals	

			final_sol = final_sol/LOOP_NUM

			if i:

				dist_est_list = []
				dist_est1 = np.linalg.norm(trocar[1]-final_sol)
				dist_est2 = np.linalg.norm(trocar[2]-final_sol)
				if dist_est1 < dist_est2:
					trocar_processing_index = 1
				else:
					trocar_processing_index = 2
			else: 
				trocar_processing_index = i
			
			rela_err = relative_err_calc(final_sol,trocar[trocar_processing_index])

			abs_err = abs_err_calc(final_sol,trocar[trocar_processing_index])

			eu_err = eudist_err_calc(final_sol,trocar[trocar_processing_index])


			covar = np.matrix(np.dot(a.T, a)).I
			residuals_err = residuals_err/LOOP_NUM
			# a_pinv = np.linalg.pinv(a.T)
			var_mtrx = np.dot(residuals_err,covar)/(a.shape[0]-3+1)
			diagonal = np.diagonal(var_mtrx)
			std_err = np.linalg.norm(diagonal)
			u,s,vh = np.linalg.svd(var_mtrx, full_matrices=True)
			print("Singular values: {} - {} - {}".format(s[0],s[1],s[2]))
			print("Trocar ground truth (mm) {}: {}".format(trocar_processing_index,trocar[trocar_processing_index]))
			print("Estimated trocar (mm): ",final_sol)
			# print("Relative error for X,Y,Z respectively (%): {} - {} - {}".format(rela_err[0],rela_err[1],rela_err[2]))
			# print("Absolute error for X,Y,Z respectively (mm): {} - {} - {}".format(abs_err[0],abs_err[1],abs_err[2]))
			# print("Root mean square error (mm): ",eu_err)
			# print("Covariance matrix associated to the estimated trocar: ",var_mtrx)
			# print("Standard error (mm): ",std_err)

			list_rela_err[ite,trocar_processing_index,:] = rela_err
			list_abs_err[ite,trocar_processing_index,:] = abs_err
			list_eu_err[ite,trocar_processing_index] = eu_err
			list_std_err[ite,trocar_processing_index] = std_err

		ite += 1

	#plot the result
	plt.figure(100),
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	fig.suptitle('Trocar 1')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,1], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,2], 'g-')
	axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	axs[0,0].set(xlabel='Noise variance', ylabel='Relative error (%)')
	# axs[0,0].set_title('Trocar 1')
	
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,0], 'r-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,2], 'g-')
	axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	axs[0,1].set(xlabel='Noise variance', ylabel='Absolute error (mm)')
	# axs[0,1].set_title('Trocar 1')

	axs[1,0].plot(list_noise_percentage, list_eu_err[:,0], 'r--')
	axs[1,0].legend(['RMSE'])
	axs[1,0].set(xlabel='Noise variance', ylabel='RMSE (mm)')
	# axs[0,2].set_title('Trocar 1')
	
	axs[1,1].plot(list_noise_percentage, list_std_err[:,0], 'r--')
	axs[1,1].legend(['Standard error'])
	axs[1,1].set(xlabel='Noise variance', ylabel='Standard error (mm)')
	# axs[0,3].set_title('Trocar 1')
	
	plt.figure(200),
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	fig.suptitle('Trocar 2')

	axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,1], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,2], 'g-')
	axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	axs[0,0].set(xlabel='Noise variance', ylabel='Relative error (%)')
	# axs[1,0].set_title('Trocar 2')

	axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,0], 'r-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,2], 'g-')
	axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	axs[0,1].set(xlabel='Noise variance', ylabel='Absolute error (mm)')
	# axs[1,1].set_title('Trocar 2')

	axs[1,0].plot(list_noise_percentage, list_eu_err[:,1], 'r--')
	axs[1,0].legend(['RMSE'])
	axs[1,0].set(xlabel='Noise variance', ylabel='RMSE (mm)')
	# axs[1,2].set_title('Trocar 2')

	axs[1,1].plot(list_noise_percentage, list_std_err[:,1], 'r--')
	axs[1,1].legend(['Standard error'])
	axs[1,1].set(xlabel='Noise variance', ylabel='Standard error (mm)')
	# axs[1,3].set_title('Trocar 2')
	
	plt.figure(300),
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	fig.suptitle('Trocar 3')

	axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,1], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,2], 'g-')
	axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	axs[0,0].set(xlabel='Noise variance', ylabel='Relative error (%)')
	# axs[2,0].set_title('Trocar 3')

	axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,0], 'r-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,2], 'g-')
	axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	axs[0,1].set(xlabel='Noise variance', ylabel='Absolute error (mm)')
	# axs[2,1].set_title('Trocar 3')

	axs[1,0].plot(list_noise_percentage, list_eu_err[:,2], 'r--')
	axs[1,0].legend(['RMSE'])
	axs[1,0].set(xlabel='Noise variance', ylabel='RMSE (mm)')	
	# axs[2,2].set_title('Trocar 3')

	axs[1,1].plot(list_noise_percentage, list_std_err[:,2], 'r--')
	axs[1,1].legend(['Standard error'])
	axs[1,1].set(xlabel='Noise variance', ylabel='Standard error (mm)')
	# axs[2,3].set_title('Trocar 3')

	plt.show()

def run_algo5(trocar, percentage):

	# Generate lines to each trocar

	vect_end = np.empty((0,3),dtype=np.float32)	
	vect_start = np.empty((0,3),dtype=np.float32)	
	N_lines = 1000
	list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)

	num_trocar = trocar.shape[0]

	for i in range(num_trocar):

		end_temp, start_temp,_,_ = generate_perfect_data(int(N_lines*percentage[i]), trocar[i]) 
		vect_end = np.append(vect_end,end_temp,axis=0)
		vect_start = np.append(vect_start,start_temp,axis=0)

	# vect_end_with_noise = add_gaussian_noise(vect_end, percentage=percentage[-1])

	outlier_end, outlier_start,_ = generate_outliers(int(N_lines*percentage[-1]), trocar[0])
	
	list_rela_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_abs_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_eu_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)
	list_std_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)

	ite = 0
	
	# for num in list_noise_percentage:



	# outlier_end_noise = add_gaussian_noise(outlier_end,var=num,percentage=1)

	vect_end =np.append(vect_end,outlier_end,axis=0)
	vect_start= np.append(vect_start,outlier_start,axis=0)

	
	# print(vect_end.shape)
	# print(vect_start.shape)
	# vect_end,vect_start = shuffle(vect_end,vect_start)

	# Define Ransac params
	P_min = 0.9999
	sample_size = 2
	
	vect_clustered = [[] for i in range(num_trocar)]
	# vect_start_clustered = [[]]*num_trocar



	temp_per = 0
	list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
	remove_idx = []
	for i in range(num_trocar):		

		list_idx_copy = list_idx.copy()

		P_outlier = 1 - percentage[i]/(1-temp_per)

		N_trial = int(math.log(1-P_min)/math.log(1-(1-P_outlier)**sample_size))
		count = 0
		N_data = int(N_lines*percentage[i])
		# print(N_data)

		# while(count < N_data):

		min_list_idx = np.array([], dtype=np.uint8).reshape(0,N_data)
		threshold = 10
		temp_idx = np.zeros(N_trial,dtype=np.uint8)
		###
		# while (threshold > 0.01):
		
		dist = []
		# print(N_trial)
		for j in range(N_trial):

			idx1 = random.choice(list_idx)
			list_idx = np.delete(list_idx,np.where(list_idx==idx1))
			
			idx2 = random.choice(list_idx)
			list_idx = np.delete(list_idx,np.where(list_idx==idx2))

			temp_idx = np.append(temp_idx,idx1)
			temp_idx = np.append(temp_idx,idx2)

			estim_pt = find_intersection_3d_lines(vect_end[idx1],vect_start[idx1],vect_end[idx2],vect_start[idx2])

			dist_temp, min_list_idx_temp = lineseg_dist(estim_pt,vect_start,vect_end,index = N_data, list_idx_lines = list_idx_copy)

			# print(len(np.unique(min_list_idx_temp)))
			min_list_idx = np.vstack([min_list_idx,min_list_idx_temp])
			
			dist.append(dist_temp)

		idx_min = dist.index(min(dist))

		threshold = dist[idx_min]

			# list_idx = np.append(list_idx,temp_idx)
			
			# list_idx = shuffle(list_idx)
		###
		# print(dist)
		# print(idx_min)
		# print(min_list_idx)
		remove_idx.append(min_list_idx[idx_min])

		vect_clustered[i] = np.copy(min_list_idx[idx_min])
		# print(np.sort(np.unique(vect_clustered[i])))
		list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
		flat_list = [item for sublist in remove_idx for item in sublist]
		flat_list = np.array(flat_list)
		# print(sorted(np.unique(flat_list)))
		list_idx = list_idx[~np.isin(list_idx,flat_list)]
		list_idx = shuffle(list_idx)
		temp_per += percentage[i]
		# print(sorted(np.unique(list_idx)))
		# print(np.sort(list_idx))
		# print(list_idx)
		# print(len(list_idx))
		# print(len(flat_list))
		# print(len(np.unique(min_list_idx[idx_min])))
		# print(list_idx)
	# for i in range(num_trocar):
	# 	print(vect_clustered[i])

	for i in range(num_trocar):

		vect_start_clustered = np.zeros((len(vect_clustered[i]),3),dtype=np.float32)
		vect_end_clustered = np.zeros((len(vect_clustered[i]),3),dtype=np.float32)
		
		for j in range(len(vect_clustered[i])):

			vect_start_clustered[j] = vect_start[vect_clustered[i][j]]
			vect_end_clustered[j] = vect_end[vect_clustered[i][j]]

		vect_rand_clustered = vect_end_clustered - vect_start_clustered

		a,b = generate_coef(vect_rand_clustered, vect_end_clustered)

		final_sol = 0
		residuals_err = 0

		for k in range(LOOP_NUM):

			x,residuals = linear_least_squares(a,b,residuals=True)
			# x,residuals = np.linalg.lstsq(a,b,rcond=None)
			final_sol += x
			residuals_err += residuals	

		final_sol = final_sol/LOOP_NUM

		if i:

			dist_est_list = []
			dist_est1 = np.linalg.norm(trocar[1]-final_sol)
			dist_est2 = np.linalg.norm(trocar[2]-final_sol)
			if dist_est1 < dist_est2:
				trocar_processing_index = 1
			else:
				trocar_processing_index = 2
		else: 
			trocar_processing_index = i
		rela_err = relative_err_calc(final_sol,trocar[trocar_processing_index])

		abs_err = abs_err_calc(final_sol,trocar[trocar_processing_index])

		eu_err = eudist_err_calc(final_sol,trocar[trocar_processing_index])


		covar = np.matrix(np.dot(a.T, a)).I
		residuals_err = residuals_err/LOOP_NUM
		# a_pinv = np.linalg.pinv(a.T)
		var_mtrx = np.dot(residuals_err,covar)/(a.shape[0]-3+1)
		diagonal = np.diagonal(var_mtrx)
		std_err = np.linalg.norm(diagonal)
		u,s,vh = np.linalg.svd(var_mtrx, full_matrices=True)
		print("Singular values: {} - {} - {}".format(s[0],s[1],s[2]))

		print("Trocar ground truth (mm) {}: {}".format(trocar_processing_index,trocar[trocar_processing_index]))
		print("Estimated trocar (mm): ",final_sol)
		DrawConfidenceRegion(s,final_sol,vh)
		print("Relative error for X,Y,Z respectively (%): {} - {} - {}".format(rela_err[0],rela_err[1],rela_err[2]))
		print("Absolute error for X,Y,Z respectively (mm): {} - {} - {}".format(abs_err[0],abs_err[1],abs_err[2]))
		print("Root mean square error (mm): ",eu_err)
		print("Covariance matrix associated to the estimated trocar: ",var_mtrx)
		print("Standard error (mm): ",std_err)

			# list_rela_err[ite,i,:] = rela_err
			# list_abs_err[ite,i,:] = abs_err
			# list_eu_err[ite,i] = eu_err
			# list_std_err[ite,i] = std_err

	# 	ite += 1

	# fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
	# ax = fig.add_subplot(111, projection='3d')

	# coefs = (1, 2, 2)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
	# # Radii corresponding to the coefficients:
	# rx, ry, rz = 1/np.sqrt(coefs)

	# # Set of all spherical angles:
	# u = np.linspace(0, 2 * np.pi, 100)
	# v = np.linspace(0, np.pi, 100)

	# # Cartesian coordinates that correspond to the spherical angles:
	# # (this is the equation of an ellipsoid):
	# x = rx * np.outer(np.cos(u), np.sin(v))
	# y = ry * np.outer(np.sin(u), np.sin(v))
	# z = rz * np.outer(np.ones_like(u), np.cos(v))

	# # Plot:
	# ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')

	# # Adjustment of the axes, so that they all have the same span:
	# max_radius = max(rx, ry, rz)
	# for axis in 'xyz':
	#     getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

	# plt.show()
	# #plot the result
	# plt.figure(100),
	# fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	# fig.suptitle('Trocar 1')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,0], 'r-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,1], 'b-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,2], 'g-')
	# axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	# axs[0,0].set(xlabel='Noise variance', ylabel='Relative error (%)')
	# # axs[0,0].set_title('Trocar 1')
	
	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,0], 'r-')
	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,1], 'b-')
	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,2], 'g-')
	# axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	# axs[0,1].set(xlabel='Noise variance', ylabel='Absolute error (mm)')
	# # axs[0,1].set_title('Trocar 1')

	# axs[1,0].plot(list_noise_percentage, list_eu_err[:,0], 'r--')
	# axs[1,0].legend(['RMSE'])
	# axs[1,0].set(xlabel='Noise variance', ylabel='RMSE (mm)')
	# # axs[0,2].set_title('Trocar 1')
	
	# axs[1,1].plot(list_noise_percentage, list_std_err[:,0], 'r--')
	# axs[1,1].legend(['Standard error'])
	# axs[1,1].set(xlabel='Noise variance', ylabel='Standard error (mm)')
	# # axs[0,3].set_title('Trocar 1')
	
	# plt.figure(200),
	# fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	# fig.suptitle('Trocar 2')

	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,0], 'r-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,1], 'b-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,2], 'g-')
	# axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	# axs[0,0].set(xlabel='Noise variance', ylabel='Relative error (%)')
	# # axs[1,0].set_title('Trocar 2')

	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,0], 'r-')
	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,1], 'b-')
	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,2], 'g-')
	# axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	# axs[0,1].set(xlabel='Noise variance', ylabel='Absolute error (mm)')
	# # axs[1,1].set_title('Trocar 2')

	# axs[1,0].plot(list_noise_percentage, list_eu_err[:,1], 'r--')
	# axs[1,0].legend(['RMSE'])
	# axs[1,0].set(xlabel='Noise variance', ylabel='RMSE (mm)')
	# # axs[1,2].set_title('Trocar 2')

	# axs[1,1].plot(list_noise_percentage, list_std_err[:,1], 'r--')
	# axs[1,1].legend(['Standard error'])
	# axs[1,1].set(xlabel='Noise variance', ylabel='Standard error (mm)')
	# # axs[1,3].set_title('Trocar 2')
	
	# plt.figure(300),
	# fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	# fig.suptitle('Trocar 3')

	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,0], 'r-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,1], 'b-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,2], 'g-')
	# axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	# axs[0,0].set(xlabel='Noise variance', ylabel='Relative error (%)')
	# # axs[2,0].set_title('Trocar 3')

	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,0], 'r-')
	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,1], 'b-')
	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,2], 'g-')
	# axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	# axs[0,1].set(xlabel='Noise variance', ylabel='Absolute error (mm)')
	# # axs[2,1].set_title('Trocar 3')

	# axs[1,0].plot(list_noise_percentage, list_eu_err[:,2], 'r--')
	# axs[1,0].legend(['RMSE'])
	# axs[1,0].set(xlabel='Noise variance', ylabel='RMSE (mm)')	
	# # axs[2,2].set_title('Trocar 3')

	# axs[1,1].plot(list_noise_percentage, list_std_err[:,2], 'r--')
	# axs[1,1].legend(['Standard error'])
	# axs[1,1].set(xlabel='Noise variance', ylabel='Standard error (mm)')
	# # axs[2,3].set_title('Trocar 3')

	# plt.show()	

def run_algo6(trocar, percentage):

	# Generate lines to each trocar

	vect_end = np.empty((0,3),dtype=np.float32)	
	vect_start = np.empty((0,3),dtype=np.float32)	
	N_lines = 1000
	list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)

	num_trocar = trocar.shape[0]

	list_rela_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_abs_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_eu_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)
	list_std_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)

	ite = 0
	count_idx = 0

	vect_end_temp = [[] for i in range(num_trocar)]
	vect_start_temp = [[] for i in range(num_trocar)]
	
	for i in range(num_trocar):

		end_temp, start_temp,_,_ = generate_perfect_data(int(N_lines*percentage[i]), trocar[i])
		vect_end_temp[i] = np.copy(end_temp)
		vect_start_temp[i] = np.copy(start_temp)
	


	for num in range(len(list_noise_percentage)):

		vect_end = np.empty((0,3),dtype=np.float32)	
		vect_start = np.empty((0,3),dtype=np.float32)	
		
		if num:

			# if count_idx == 2:
			# 	count_idx = 0

			# percentage[-1] = num/100
			# percentage[count_idx] = percentage[count_idx]-step/100
			# count_idx += 1


			#Update the percentage ratio for RANSAC
			percentage[0] -= 0.03
			percentage[1] -= 0.01
			percentage[2] -= 0.01
			
			for i in range(num_trocar):

				if i:

					# outlier_end, outlier_start,_ = generate_outliers(int(N_lines*0.03), trocar[i])
					# end_temp, start_temp,_,_ = generate_perfect_data(int(N_lines*percentage[i]), trocar[i]) 
					vect_end_with_noise = add_gaussian_noise(vect_end_temp[i], var=50,percentage=0.01*ite)

				else:
			
					# end_temp, start_temp,_,_ = generate_perfect_data(int(N_lines*percentage[i]), trocar[i]) 
					# outlier_end, outlier_start,_ = generate_outliers(int(N_lines*0.01), trocar[i])
					vect_end_with_noise = add_gaussian_noise(vect_end_temp[i], var=50,percentage=0.03*ite)

				vect_end =np.append(vect_end,vect_end_with_noise,axis=0)
				vect_start= np.append(vect_start,vect_start_temp[i],axis=0)



			vect_end,vect_start = shuffle(vect_end,vect_start)

			# vect_end_with_noise = add_gaussian_noise(vect_end, percentage=percentage[-1])

			

			
			# for num in list_noise_percentage:



			# outlier_end_noise = add_gaussian_noise(outlier_end,var=num,percentage=1)


	
		# print(vect_end.shape)
		# print(vect_start.shape)
		# vect_end,vect_start = shuffle(vect_end,vect_start)
		# if num:
			# print("Iteration: ",ite)
			# print(num)
			# Define Ransac params
			P_min = 0.99
			sample_size = 2
			
			vect_clustered = [[] for i in range(num_trocar)]
			# vect_start_clustered = [[]]*num_trocar



			temp_per = 0
			list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
			remove_idx = []
			for i in range(num_trocar):		

				list_idx_copy = list_idx.copy()

				P_outlier = 1 - percentage[i]/(1-temp_per)
				# print(P_outlier)
				N_trial = int(math.log(1-P_min)/math.log(1-(1-P_outlier)**sample_size))
				
				# if N_trial > len(list_idx):
				# 	N_trial = len(list_idx)

				count = 0
				N_data = int(N_lines*percentage[i])
				# print(N_data)

				# while(count < N_data):

				min_list_idx = np.array([], dtype=np.uint8).reshape(0,N_data)
				threshold = 10
				# temp_idx = np.zeros(N_trial,dtype=np.uint8)
				###
				# while (threshold > 0.01):
				
				dist = []
				# print(N_trial)
				for j in range(N_trial):

					idx1 = random.choice(list_idx)
					list_idx = np.delete(list_idx,np.where(list_idx==idx1))
					
					idx2 = random.choice(list_idx)
					list_idx = np.delete(list_idx,np.where(list_idx==idx2))

					# temp_idx = np.append(temp_idx,idx1)
					# temp_idx = np.append(temp_idx,idx2)

					estim_pt = find_intersection_3d_lines(vect_end[idx1],vect_start[idx1],vect_end[idx2],vect_start[idx2])

					dist_temp, min_list_idx_temp = lineseg_dist(estim_pt,vect_start,vect_end,index = N_data, list_idx_lines = list_idx_copy)

					# print(len(np.unique(min_list_idx_temp)))
					min_list_idx = np.vstack([min_list_idx,min_list_idx_temp])
					
					dist.append(dist_temp)

				idx_min = dist.index(min(dist))

				threshold = dist[idx_min]

					# list_idx = np.append(list_idx,temp_idx)
					
					# list_idx = shuffle(list_idx)
				###
				# print(dist)
				# print(idx_min)
				# print(min_list_idx)
				remove_idx.append(min_list_idx[idx_min])

				vect_clustered[i] = np.copy(min_list_idx[idx_min])
				# print(np.sort(np.unique(vect_clustered[i])))
				list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
				flat_list = [item for sublist in remove_idx for item in sublist]
				flat_list = np.array(flat_list)
				# print(sorted(np.unique(flat_list)))
				list_idx = list_idx[~np.isin(list_idx,flat_list)]
				list_idx = shuffle(list_idx)
				temp_per += percentage[i]
				# print(sorted(np.unique(list_idx)))
				# print(np.sort(list_idx))
				# print(list_idx)
				# print(len(list_idx))
				# print(len(flat_list))
				# print(len(np.unique(min_list_idx[idx_min])))
				# print(list_idx)
			# for i in range(num_trocar):
			# 	print(vect_clustered[i])

		for i in range(num_trocar):

			if num:
				
				vect_start_clustered = np.zeros((len(vect_clustered[i]),3),dtype=np.float32)
				vect_end_clustered = np.zeros((len(vect_clustered[i]),3),dtype=np.float32)
				
				for j in range(len(vect_clustered[i])):

					vect_start_clustered[j] = vect_start[vect_clustered[i][j]]
					vect_end_clustered[j] = vect_end[vect_clustered[i][j]]

				vect_rand_clustered = vect_end_clustered - vect_start_clustered

				a,b = generate_coef(vect_rand_clustered, vect_end_clustered)
			
			else:

				vect_rand_clustered = vect_end_temp[i] - vect_start_temp[i]
				a,b = generate_coef(vect_rand_clustered, vect_end_temp[i])


			final_sol = 0
			residuals_err = 0

			for k in range(LOOP_NUM):

				x,residuals = linear_least_squares(a,b,residuals=True)
				# x,residuals = np.linalg.lstsq(a,b,rcond=None)
				final_sol += x
				residuals_err += residuals	

			final_sol = final_sol/LOOP_NUM

			if i:

				dist_est_list = []
				dist_est1 = np.linalg.norm(trocar[1]-final_sol)
				dist_est2 = np.linalg.norm(trocar[2]-final_sol)
				if dist_est1 < dist_est2:
					trocar_processing_index = 1
				else:
					trocar_processing_index = 2
			else: 
				trocar_processing_index = i

			rela_err = relative_err_calc(final_sol,trocar[trocar_processing_index])

			abs_err = abs_err_calc(final_sol,trocar[trocar_processing_index])

			eu_err = eudist_err_calc(final_sol,trocar[trocar_processing_index])


			covar = np.matrix(np.dot(a.T, a)).I
			residuals_err = residuals_err/LOOP_NUM
			# a_pinv = np.linalg.pinv(a.T)
			var_mtrx = np.dot(residuals_err,covar)/(a.shape[0]-3+1)
			diagonal = np.diagonal(var_mtrx)
			std_err = np.linalg.norm(diagonal)
			u,s,vh = np.linalg.svd(var_mtrx, full_matrices=True)
			print("Singular values: {} - {} - {}".format(s[0],s[1],s[2]))

			print("Trocar ground truth (mm) {}: {}".format(trocar_processing_index,trocar[trocar_processing_index]))
			print("Estimated trocar (mm): ",final_sol)
			# DrawConfidenceRegion(s,final_sol,vh)
			print("Relative error for X,Y,Z respectively (%): {} - {} - {}".format(rela_err[0],rela_err[1],rela_err[2]))
			print("Absolute error for X,Y,Z respectively (mm): {} - {} - {}".format(abs_err[0],abs_err[1],abs_err[2]))
			print("Root mean square error (mm): ",eu_err)
			print("Covariance matrix associated to the estimated trocar: ",var_mtrx)
			print("Standard error (mm): ",std_err)

			list_rela_err[ite,i,:] = rela_err
			list_abs_err[ite,i,:] = abs_err
			list_eu_err[ite,i] = eu_err
			list_std_err[ite,i] = std_err

		ite += 1


	#plot the result
	plt.figure(100),
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	fig.suptitle('Trocar 1')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,1], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,2], 'g-')
	axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	axs[0,0].set(xlabel='Incorrect data (%)', ylabel='Relative error (%)')
	# axs[0,0].set_title('Trocar 1')
	
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,0], 'r-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,2], 'g-')
	axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	axs[0,1].set(xlabel='Incorrect data (%)', ylabel='Absolute error (mm)')
	# axs[0,1].set_title('Trocar 1')

	axs[1,0].plot(list_noise_percentage, list_eu_err[:,0], 'r--')
	axs[1,0].legend(['RMSE'])
	axs[1,0].set(xlabel='Incorrect data (%)', ylabel='RMSE (mm)')
	# axs[0,2].set_title('Trocar 1')
	
	axs[1,1].plot(list_noise_percentage, list_std_err[:,0], 'r--')
	axs[1,1].legend(['Standard error'])
	axs[1,1].set(xlabel='Incorrect data (%)', ylabel='Standard error (mm)')
	# axs[0,3].set_title('Trocar 1')
	
	plt.figure(200),
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	fig.suptitle('Trocar 2')
	fig.suptitle('Trocar 2')

	axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,1], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,2], 'g-')
	axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	axs[0,0].set(xlabel='Incorrect data (%)', ylabel='Relative error (%)')
	# axs[1,0].set_title('Trocar 2')

	axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,0], 'r-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,2], 'g-')
	axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	axs[0,1].set(xlabel='Incorrect data (%)', ylabel='Absolute error (mm)')
	# axs[1,1].set_title('Trocar 2')

	axs[1,0].plot(list_noise_percentage, list_eu_err[:,1], 'r--')
	axs[1,0].legend(['RMSE'])
	axs[1,0].set(xlabel='Incorrect data (%)', ylabel='RMSE (mm)')
	# axs[1,2].set_title('Trocar 2')

	axs[1,1].plot(list_noise_percentage, list_std_err[:,1], 'r--')
	axs[1,1].legend(['Standard error'])
	axs[1,1].set(xlabel='Incorrect data (%)', ylabel='Standard error (mm)')
	# axs[1,3].set_title('Trocar 2')
	
	plt.figure(300),
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	fig.suptitle('Trocar 3')

	axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,1], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,2], 'g-')
	axs[0,0].legend(['Relative error for X','Relative error for Y','Relative error for Z'])
	axs[0,0].set(xlabel='Incorrect data (%)', ylabel='Relative error (%)')
	# axs[2,0].set_title('Trocar 3')

	axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,0], 'r-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,2], 'g-')
	axs[0,1].legend(['Absolute error for X','Absolute error for Y','Absolute error for Z'])
	axs[0,1].set(xlabel='Incorrect data (%)', ylabel='Absolute error (mm)')
	# axs[2,1].set_title('Trocar 3')

	axs[1,0].plot(list_noise_percentage, list_eu_err[:,2], 'r--')
	axs[1,0].legend(['RMSE'])
	axs[1,0].set(xlabel='Incorrect data (%)', ylabel='RMSE (mm)')	
	# axs[2,2].set_title('Trocar 3')

	axs[1,1].plot(list_noise_percentage, list_std_err[:,2], 'r--')
	axs[1,1].legend(['Standard error'])
	axs[1,1].set(xlabel='Incorrect data (%)', ylabel='Standard error (mm)')
	# axs[2,3].set_title('Trocar 3')

	plt.show()	



def EvaluateLsqSolution(covariance_mtrx, pts_estimated):
	'''
	Evaluate the estimation result using linearized statistics
	'''

def DrawConfidenceRegion(s,center,rotation):

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
	        temp = np.dot([x[i,j],y[i,j],z[i,j]],rotation) + center
	        x[i,j] = temp[0,0]
	        y[i,j] = temp[0,1]
	        z[i,j] = temp[0,2]
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
	plt.show()

def visualize_model(trocar, pts = None, vect_end = None, vect_start = None, line_idx = None):

	# Draw 3d graph

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('X (mm)')
	ax.set_ylabel('Y (mm)')
	ax.set_zlabel('Z (mm)')

	r = [-120, 120]
	for s, e in combinations(np.array(list(product(r, r, r))), 2):
		if np.sum(np.abs(s-e)) == r[1]-r[0]:
			ax.plot3D(*zip(s, e), color="#0c0c0d")


	# draw cloud points
	if pts is not None:		
		for i in range(pts.shape[0]):
			ax.scatter(pts[i,0],pts[i,1],pts[i,2],marker = ",",color="#948e8e")

	if line_idx is not None:

		#draw lines in each cluster
		cycol = cycle('grcmy')
		for i in range(len(line_idx)):
			color = next(cycol)
			for idx in line_idx[i]:
				ax.plot([vect_start[idx][0], vect_end[idx][0]], [vect_start[idx][1],vect_end[idx][1]],zs=[vect_start[idx][2],vect_end[idx][2]],color=color)

	else:
		#draw all lines
		if vect_start is not None:
			N_lines = vect_start.shape[0]
			for i in range(N_lines):
			    ax.plot([vect_start[i][0], vect_end[i][0]], [vect_start[i][1],vect_end[i][1]],zs=[vect_start[i][2],vect_end[i][2]])


	#draw trocar point
	for i in range(trocar.shape[0]):
		ax.scatter(trocar[i,0],trocar[i,1],trocar[i,2],marker = "*",color="b")

	plt.show()

def ransac_new(trocar,percentage):

	num_trocar = trocar.shape[0]

	#Split the outlier percentage to each trocar
	percentage_new = percentage[:-1]
	
	outlier_per = percentage[-1]/num_trocar
	
	temp_out = 0

	for i in range(num_trocar):

		if i == num_trocar-1:
			
			last_out = percentage[-1] - temp_out

			percentage_new = np.append(percentage_new,last_out)

		percentage_new = np.append(percentage_new,outlier_per)

		temp_out += outlier_per

	# Generate lines to each trocar
	
	vect_end = np.empty((0,3),dtype=np.float32)	
	vect_start = np.empty((0,3),dtype=np.float32)	
	N_lines = 1000
	list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
	list_idx_gt = []
	cur = 0
	last = 0
	
	for i in range(num_trocar):

		end_temp, start_temp,_,_ = generate_perfect_data(int(N_lines*percentage_new[i]), trocar[i], scale1 = SCALE_COEF, scale2 = 5) 
		vect_end = np.append(vect_end,end_temp,axis=0)
		vect_start = np.append(vect_start,start_temp,axis=0)
		cur += int(N_lines*percentage_new[i])
		# print("cur: {}, last: {}",cur,last)

		list_temp = np.arange(last, cur)

		# print("List temp: ",list_temp)

		list_idx_gt.append(list_temp)
		
		last += int(N_lines*percentage_new[i])

	# vect_end_with_noise = add_gaussian_noise(vect_end, percentage=percentage[-1])
	
	for i in range(num_trocar):

		outlier_end, outlier_start,_ = generate_outliers(int(N_lines*percentage_new[num_trocar+i]), trocar[i], scale1 = SCALE_COEF, scale2 = 5)
		vect_end = np.append(vect_end,outlier_end,axis=0)
		vect_start = np.append(vect_start,outlier_start,axis=0)
		cur += int(N_lines*percentage_new[num_trocar+i])
		# print("cur: {}, last: {}",cur,last)

		list_temp = np.arange(last, cur)

		# print("List temp: ",list_temp)

		if i == 0:

			list_idx_gt.append(list_temp)

		else:

			list_idx_gt[num_trocar] = np.append(list_idx_gt[num_trocar],list_temp)

		last += int(N_lines*percentage_new[num_trocar+i])

	list_rela_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_abs_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_eu_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)
	list_std_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)

	# outlier_end_noise = add_gaussian_noise(outlier_end,var=num,percentage=1)

	vect_end =np.append(vect_end,outlier_end,axis=0)
	vect_start= np.append(vect_start,outlier_start,axis=0)

	num_trials = 100000000
	sample_count = 0
	sample_size = 2
	P_min = 0.99
	temp_per = 0

	list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
	# remove_idx = []
	vect_clustered = []
	threshold_dist = 1
	threshold_inliers = 10

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
		#update RANSAC params
		if num_inliers:

			P_outlier = 1 - num_inliers/(N_lines-temp_per)
			# print(P_outlier)
			num_trials = int(math.log(1-P_min)/math.log(1-(1-P_outlier)**sample_size))

		if num_inliers > threshold_inliers:

			# remove_idx.append(min_list_idx_temp)

			vect_clustered.append(min_list_idx_temp)
			list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
			flat_list = [item for sublist in vect_clustered for item in sublist]
			flat_list = np.array(flat_list)
			# print(sorted(np.unique(flat_list)))
			list_idx = list_idx[~np.isin(list_idx,flat_list)]
			
			if len(list_idx) < 2:

				vect_clustered.append(list_idx)
				list_idx = []
				break

			list_idx = shuffle(list_idx)

			#reset RANSAC params
			sample_count = 0
			num_trials = 100000000
			temp_per += num_inliers
			
			print(num_inliers)

	#Store the last cluster (if any)
	if len(list_idx):

		vect_clustered.append(list_idx)
			 
	if len(vect_clustered) == num_trocar+1:

		for i in range(num_trocar):

			vect_clustered[i] = sorted(vect_clustered[i])

			list_idx_gt[i] = sorted(list_idx_gt[i])

		y_true = sorted(list_idx_gt)
		y_pred = sorted(vect_clustered)

		for i in range(num_trocar):
			print(y_true[i])
			print(y_pred[i])
			
		# cm = confusion_matrix(y_true,y_pred,)
		# disp = ConfusionMatrixDisplay(confusion_matrix=cm,
  #                             display_labels=display_labels)


		# # NOTE: Fill all variables here with default values of the plot_confusion_matrix
		# disp = disp.plot(include_values=include_values,
		#                  cmap=cmap, ax=ax, xticks_rotation=xticks_rotation)

		# plt.show()
		# print(sorted(vect_clustered) == sorted(list_idx_gt))


	else:

		print("Wrongly classify")


	# plydata = PlyData.read("liver_simplified.ply")
	# vertex_data = plydata['vertex'].data # numpy array with fields ['x', 'y', 'z']
	# pts = np.zeros([vertex_data.size, 3])
	# pts[:, 0] = vertex_data['x']
	# pts[:, 1] = vertex_data['y']
	# pts[:, 2] = vertex_data['z']

	# visualize_model(trocar=trocar,vect_end=vect_end,vect_start=vect_start,line_idx=list_idx_gt)
	
	# visualize_model(trocar=trocar,vect_end=vect_end,vect_start=vect_start,line_idx=vect_clustered)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

###################################################################

if __name__ == '__main__':

	trocar = np.array([[-60,70,120],[-65,-70,120],[65,-75,120],[60,75,120]])
	# trocar = np.array([[30,68,125],[150,70,130],[35, 200,120]])
	# percentage = np.array([0.5,0.3,0.2,0.1])
	# run_algo6(trocar,percentage)

	# percentage = np.array([0.35,0.27,0.18,0.2])
	# run_algo4(trocar,percentage)

	percentage = np.array([0.35,0.25,0.15,0.1,0.15])
	# run_algo5(trocar,percentage)
	
	ransac_new(trocar,percentage)
