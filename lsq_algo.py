import numpy as np 
import shapely
from generate_data import *
from scipy.optimize import least_squares
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle

LOOP_NUM = 10
N_lines = 1000
# percentage = 0.2
# num_outliers = int(N_lines*percentage)

start_range = 0
end_range = 50
step = (end_range - start_range)/10


def lineseg_dist(p, a, b, index = 0):

	dist = []
	dist_val = 0
	list_idx = []

	for i in range(len(a)):

		x = a[i] - b[i]
		t = np.dot(p-b[i], x)/np.dot(x, x)
		dist_val = np.linalg.norm(t*(a[i]-b[i])+b[i]-p)
		dist.append(dist_val)
	
	if index:

		# dist = dist.sort()
		# dist = dist[0:index]
		max_dist = max(dist)
		for i in range(index):

			temp_idx = dist.index(min(dist))
			list_idx.append(temp_idx)
			dist[temp_idx] = max_dist
		return np.linalg.norm(dist), np.asarray(list_idx,dtype=np.float32)

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
    
    if diff < 0.01:

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
	P_min = 0.99
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
		min_list_idx = np.zeros((N_trial,N_data),dtype=np.float32)

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
		list_idx = np.delete(list_idx, np.asarray(remove_idx,dtype=np.uint8).flatten())
		list_idx = shuffle(list_idx)

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
		a_pinv = np.linalg.pinv(a.T)
		var_mtrx = np.dot(residuals_err,a_pinv)/(a.shape[0]-3+1)
		diagonal = np.diagonal(var_mtrx)
		std_err = np.linalg.norm(diagonal)
		
		print("Trocar ground truth {}: {}".format(i,trocar[i]))
		print("Estimated trocar: ",final_sol)
		print("Relative error for X,Y,Z respectively (%): {} - {} - {}".format(rela_err[0],rela_err[1],rela_err[2]))
		print("Absolute error for X,Y,Z respectively: {} - {} - {}".format(abs_err[0],abs_err[1],abs_err[2]))
		print("Root mean square error: ",eu_err)
		print("Standard error: ",std_err)





				# error = np.linalg.norm(trocar[i]-estim_pt)

				# list_error.append(error)



###################################################################



if __name__ == '__main__':

	trocar_c = np.array([[30,68,125],[150,70,130],[35, 200,120]])
	percentage = np.array([0.5,0.2,0.2,0.1])
	run_algo3(trocar_c,percentage)
	# random_unit_vector()