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
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from pyntcloud import PyntCloud
from plyfile import PlyData
from itertools import product, combinations, cycle
import time

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


def ransac_new(trocar,percentage):

	num_trocar = trocar.shape[0]

	#Split the outlier percentage to each trocar
	percentage_new = percentage[:-1]

	# Generate lines to each trocar
	
	# vect_end_temp = np.empty((0,3),dtype=np.float32)	
	# vect_start_temp = np.empty((0,3),dtype=np.float32)
	vect_end = np.empty((0,3),dtype=np.float32)	
	vect_start = np.empty((0,3),dtype=np.float32)	
	N_lines = 1000
	list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
	list_idx_gt = []
	cur = 0
	last = 0
	dict_gt = {}
	dict_cluster = {}

	for i in range(num_trocar):

		end_temp, start_temp,_,_ = generate_data(int(N_lines*percentage_new[i]), trocar[i], scale1 = SCALE_COEF1, scale2 = SCALE_COEF2) 
		vect_end = np.append(vect_end,end_temp,axis=0)
		vect_start = np.append(vect_start,start_temp,axis=0)
		cur += int(N_lines*percentage_new[i])
		# print("cur: {}, last: {}",cur,last)

		list_temp = np.arange(last, cur)

		# print("List temp: ",list_temp)

		list_idx_gt.append(list_temp)
		
		last += int(N_lines*percentage_new[i])

	### add Incorrect data
	if percentage[-1]:

		# vect_end,vect_start, random_list = add_gaussian_noise(vect_end,vect_start, sigma=40, percentage=percentage[-1])
	
		# for i in range(len(list_idx_gt)):

		# 	list_idx_gt[i] = list_idx_gt[i][~np.isin(list_idx_gt[i],random_list)]
		# 	list_idx_gt[i] = list_idx_gt[i].tolist()

		# list_idx_gt.append(random_list.tolist())

		outlier_per = percentage[-1]/num_trocar
		
		temp_out = 0

		for i in range(num_trocar):

			if i == num_trocar-1:
				
				last_out = percentage[-1] - temp_out

				percentage_new = np.append(percentage_new,last_out)

			percentage_new = np.append(percentage_new,outlier_per)

			temp_out += outlier_per

		for i in range(num_trocar):

			outlier_end, outlier_start,_,_ = generate_incorrect_data(int(N_lines*percentage_new[num_trocar+i]), trocar[i], scale1 = SCALE_COEF1, scale2 = SCALE_COEF2)
			vect_end = np.append(vect_end,outlier_end,axis=0)
			vect_start = np.append(vect_start,outlier_start,axis=0)
			cur += int(N_lines*percentage_new[num_trocar+i])
			print("cur: {}, last: {}",cur,last)

			list_temp = np.arange(last, cur)

			print("List temp: ",list_temp)

			if i == 0:

				list_idx_gt.append(list_temp.tolist())

			else:

				list_idx_gt[num_trocar].extend(list_temp.tolist())

			last += int(N_lines*percentage_new[num_trocar+i])


	for i in range(len(list_idx_gt)):

		if i == num_trocar:

			dict_gt["Incorrect Data"] = list_idx_gt[i]
		
		else:

			dict_gt["Trocar "+str(i+1)] = list_idx_gt[i]
			
	# print(dict_gt)
	### add noise by distance
	# if percentage[-1]:

	# 	outlier_per = percentage[-1]/num_trocar
		
	# 	temp_out = 0

	# 	for i in range(num_trocar):

	# 		if i == num_trocar-1:
				
	# 			last_out = percentage[-1] - temp_out

	# 			percentage_new = np.append(percentage_new,last_out)

	# 		percentage_new = np.append(percentage_new,outlier_per)

	# 		temp_out += outlier_per

		# for i in range(num_trocar):

			# outlier_end, outlier_start,_ = generate_outliers(int(N_lines*percentage_new[num_trocar+i]), trocar[i], scale1 = SCALE_COEF1, scale2 = SCALE_COEF2)
			# vect_end = np.append(vect_end,outlier_end,axis=0)
			# vect_start = np.append(vect_start,outlier_start,axis=0)
			# cur += int(N_lines*percentage_new[num_trocar+i])
			# print("cur: {}, last: {}",cur,last)

			# list_temp = np.arange(last, cur)

			# print("List temp: ",list_temp)

			# if i == 0:

				# list_idx_gt.append(list_temp.tolist())

			# else:

				# list_idx_gt[num_trocar].extend(list_temp.tolist())

			# last += int(N_lines*percentage_new[num_trocar+i])

	list_rela_err = np.zeros((num_trocar,3),dtype=np.float32)
	list_abs_err = np.zeros((num_trocar,3),dtype=np.float32)
	list_eu_err = np.zeros((num_trocar,1),dtype=np.float32)
	list_std_err = np.zeros((num_trocar,1),dtype=np.float32)

	# num_trials = 100000000
	# sample_count = 0
	# sample_size = 2
	# P_min = 0.99
	# temp_per = 0

	# list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
	# # remove_idx = []
	# vect_clustered = []
	length_clus = num_trocar + 4
	list_idx = np.random.choice(N_lines, size=N_lines, replace=False)

	# while(length_clus > num_trocar+1):

	# 	num_trials = 100000000
	# 	sample_count = 0
	# 	sample_size = 2
	# 	P_min = 0.99
	# 	temp_per = 0

	# 	list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
	# 	# remove_idx = []
	# 	vect_clustered = []
	# 	threshold_dist = 1
	# 	threshold_inliers = 20
	# 	# start_time = time.time()
	# 	while(num_trials > sample_count):
			
	# 		sample_count += 1
			
	# 		list_idx_copy = list_idx.copy()
			
	# 		idx1 = random.choice(list_idx)
	# 		# list_idx = np.delete(list_idx,np.where(list_idx==idx1))

	# 		idx2 = random.choice(list_idx)
			
	# 		# while(idx2 == idx1):
				
	# 		# 	idx2 = random.choice(list_idx)
	# 		# list_idx = np.delete(list_idx,np.where(list_idx==idx2))

	# 		estim_pt = find_intersection_3d_lines(vect_end[idx1], vect_start[idx1], vect_end[idx2], vect_start[idx2])
			
	# 		min_list_idx_temp = lineseg_dist(estim_pt, vect_start, vect_end, list_idx_lines = list_idx_copy, threshold = threshold_dist)

	# 		num_inliers = len(min_list_idx_temp)

	# 		#update RANSAC params
	# 		if num_inliers:

	# 			P_outlier = 1 - num_inliers/(N_lines-temp_per)
				
	# 			if not P_outlier:

	# 				vect_clustered.append(list_idx.tolist())
	# 				list_idx = []
	# 				break


	# 			num_trials = int(math.log(1-P_min)/math.log(1-(1-P_outlier)**sample_size))

	# 		if num_inliers > threshold_inliers:

	# 			# remove_idx.append(min_list_idx_temp)

	# 			vect_clustered.append(min_list_idx_temp.tolist())
	# 			list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
	# 			flat_list = [item for sublist in vect_clustered for item in sublist]
	# 			flat_list = np.array(flat_list)
	# 			# print(sorted(np.unique(flat_list)))
	# 			list_idx = list_idx[~np.isin(list_idx,flat_list)]
				
	# 			if not len(list_idx):

	# 				break

	# 			elif len(list_idx) < 3:

	# 				vect_clustered.append(list_idx.tolist())
	# 				list_idx = []
	# 				break

	# 			list_idx = shuffle(list_idx)

	# 			#reset RANSAC params
	# 			sample_count = 0
	# 			num_trials = 100000000
	# 			temp_per += num_inliers
	# 			# print(num_inliers)		
	# 	#Store the last cluster (if any)
	# 	if len(list_idx):

	# 		vect_clustered.append(list_idx.tolist())
		
	# 	length_clus = len(vect_clustered)


	# # print("--- %s seconds ---" % (time.time() - start_time))
	# if len(vect_clustered) < num_trocar+2:

	# 	for key,value in dict_gt.items():

	# 		gt_temp = set(value)
	# 		match_max = 0
	# 		match_index = 0

	# 		for i in range(len(vect_clustered)):

	# 			vect_temp = set(vect_clustered[i])

	# 			match = len(gt_temp & vect_temp)

	# 			if match > match_max:

	# 				match_max = match
	# 				match_index = i

	# 		dict_cluster[key]  = vect_clustered[match_index]

	# 	y_true = np.arange(N_lines)
	# 	y_true = y_true.tolist()

	# 	y_pred = np.arange(N_lines)
	# 	y_pred = y_pred.tolist()

	# 	for key,value in dict_gt.items():

	# 		for val in value:

	# 			y_true[val] = key

	# 	for key,value in dict_cluster.items():

	# 		for val in value:

	# 			y_pred[val] = key

	# 	# plot_cfs_matrix(y_true,y_pred,list(dict_gt.keys()))


	# else:
	# 	print(len(vect_clustered))
	# 	print("Wrongly classify")
	# 	return

	# ite = 0

	# for key,value in dict_cluster.items():

	# 	if ite == num_trocar:

	# 		break

	# 	vect_start_clustered = np.zeros((len(value),3),dtype=np.float32)
	# 	vect_end_clustered = np.zeros((len(value),3),dtype=np.float32)	

	# 	vect_start_clustered = vect_start[value]
	# 	vect_end_clustered = vect_end[value]

	# 	vect_rand_clustered = (vect_end_clustered - vect_start_clustered)/(SCALE_COEF1*(SCALE_COEF2+1))

	# 	a,b = generate_coef(vect_rand_clustered, vect_end_clustered)

	# 	final_sol,residuals_err = linear_least_squares(a,b,residuals=True)

	# 	rela_err = relative_err_calc(final_sol,trocar[ite])

	# 	abs_err = abs_err_calc(final_sol,trocar[ite])

	# 	eu_err = eudist_err_calc(final_sol,trocar[ite])
		
	# 	# covar = np.matrix(np.dot(a.T, a)).I
	# 	covar = np.linalg.pinv(np.dot(a.T, a))
	# 	var_mtrx = np.dot(residuals_err,covar)/(a.shape[0]-3+1)
	# 	diagonal = np.diagonal(var_mtrx)
	# 	std_err = np.linalg.norm(diagonal)
	# 	u,s,vh = np.linalg.svd(var_mtrx, full_matrices=True)
	# 	print("Estimated trocar (mm): ",final_sol)
	# 	print("{} ground truth (mm): {}".format(key,trocar[ite]))
	# 	print("Singular values: {} - {} - {}".format(s[0],s[1],s[2]))
	# 	print("Standard error (mm): ",std_err)
	# 	# print("Estimated trocar (mm): ",final_sol)
	# 	# DrawConfidenceRegion(s,final_sol,vh)
	# 	print("Relative error for X,Y,Z respectively (%): {} - {} - {}".format(rela_err[0],rela_err[1],rela_err[2]))
	# 	print("Absolute error for X,Y,Z respectively (mm): {} - {} - {}".format(abs_err[0],abs_err[1],abs_err[2]))
	# 	print("Root mean square error (mm): ",eu_err)
	# 	# DrawConfidenceRegion(s,final_sol,vh,key)	
	# 	list_rela_err[ite,:] = rela_err
	# 	list_abs_err[ite,:] = abs_err
	# 	list_eu_err[ite] = eu_err
	# 	list_std_err[ite] = std_err
	# 	ite += 1


	plydata = PlyData.read("liver_simplified.ply")
	vertex_data = plydata['vertex'].data # numpy array with fields ['x', 'y', 'z']
	pts = np.zeros([vertex_data.size, 3])
	pts[:, 0] = vertex_data['x']
	pts[:, 1] = vertex_data['y']
	pts[:, 2] = vertex_data['z']

	visualize_model(pts=pts,trocar=trocar,vect_end=vect_end,vect_start=vect_start,line_idx=dict_gt,gt=True)
	
	# visualize_model(trocar=trocar,vect_end=vect_end,vect_start=vect_start,line_idx=dict_cluster,gt=False)
	# visualize_model(pts=pts)
	# return list_rela_err, list_abs_err, list_eu_err, list_std_err

def test_case(trocar, percentage):

	num_trocar = trocar.shape[0]

	#Split the outlier percentage to each trocar
	percentage_new = percentage[:-1]

	# Generate lines to each trocar
	
	# vect_end_temp = np.empty((0,3),dtype=np.float32)	
	# vect_start_temp = np.empty((0,3),dtype=np.float32)
	vect_end = np.empty((0,3),dtype=np.float32)	
	vect_start = np.empty((0,3),dtype=np.float32)	
	N_lines = 1000
	list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
	list_idx_gt = []
	cur = 0
	last = 0


	for i in range(num_trocar):

		end_temp, start_temp,_,_ = generate_perfect_data(int(N_lines*percentage_new[i]), trocar[i], scale1 = SCALE_COEF1, scale2 = SCALE_COEF2) 
		vect_end = np.append(vect_end,end_temp,axis=0)
		vect_start = np.append(vect_start,start_temp,axis=0)
		cur += int(N_lines*percentage_new[i])
		# print("cur: {}, last: {}",cur,last)

		list_temp = np.arange(last, cur)

		# print("List temp: ",list_temp)

		list_idx_gt.append(list_temp)
		
		last += int(N_lines*percentage_new[i])

	list_rela_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_abs_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	list_eu_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)
	list_std_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)
	counter = 0
	for num in list_noise_percentage:

		list_idx_gt_with_noise = list_idx_gt.copy()
		dict_gt = {}
		dict_cluster = {}
		### add Gaussian noise
		if num:

			vect_end_with_noise,vect_start_with_noise, random_list = add_gaussian_noise(vect_end,vect_start, sigma=20, percentage=num/100)
		
			for i in range(len(list_idx_gt_with_noise)):

				list_idx_gt_with_noise[i] = list_idx_gt_with_noise[i][~np.isin(list_idx_gt_with_noise[i],random_list)]
				list_idx_gt_with_noise[i] = list_idx_gt_with_noise[i].tolist()

			list_idx_gt_with_noise.append(random_list.tolist())
		else:
			vect_end_with_noise = np.copy(vect_end)
			vect_start_with_noise = np.copy(vect_start)
		for i in range(len(list_idx_gt_with_noise)):

			if i == num_trocar:

				dict_gt["Incorrect Data"] = list_idx_gt_with_noise[i]
			
			else:

				dict_gt["Trocar "+str(i+1)] = list_idx_gt_with_noise[i]

		length_clus = num_trocar + 4
		list_idx = np.random.choice(N_lines, size=N_lines, replace=False)

		while(length_clus > num_trocar+2):

			num_trials = 100000000
			sample_count = 0
			sample_size = 2
			P_min = 0.99
			temp_per = 0

			list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
			# remove_idx = []
			vect_clustered = []
			threshold_dist = 1
			threshold_inliers = 20
			# start_time = time.time()


			while(num_trials > sample_count):
				
				sample_count += 1
				
				list_idx_copy = list_idx.copy()
				
				idx1 = random.choice(list_idx)
				# list_idx = np.delete(list_idx,np.where(list_idx==idx1))

				idx2 = random.choice(list_idx)
				
				# while(idx2 == idx1):
					
				# 	idx2 = random.choice(list_idx)
				# list_idx = np.delete(list_idx,np.where(list_idx==idx2))

				estim_pt = find_intersection_3d_lines(vect_end_with_noise[idx1], vect_start_with_noise[idx1], vect_end_with_noise[idx2], vect_start_with_noise[idx2])
				
				min_list_idx_temp = lineseg_dist(estim_pt, vect_start_with_noise, vect_end_with_noise, list_idx_lines = list_idx_copy, threshold = threshold_dist)

				num_inliers = len(min_list_idx_temp)

				#update RANSAC params
				if num_inliers:

					P_outlier = 1 - num_inliers/(N_lines-temp_per)
					
					if not P_outlier:

						vect_clustered.append(list_idx.tolist())
						list_idx = []
						break


					num_trials = int(math.log(1-P_min)/math.log(1-(1-P_outlier)**sample_size))

				if num_inliers > threshold_inliers:

					# remove_idx.append(min_list_idx_temp)

					vect_clustered.append(min_list_idx_temp.tolist())
					list_idx = np.random.choice(N_lines, size=N_lines, replace=False)
					flat_list = [item for sublist in vect_clustered for item in sublist]
					flat_list = np.array(flat_list)
					# print(sorted(np.unique(flat_list)))
					list_idx = list_idx[~np.isin(list_idx,flat_list)]
					
					if not len(list_idx):

						break

					elif len(list_idx) < 3:

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

				vect_clustered.append(list_idx.tolist())	

			length_clus = len(vect_clustered)


		# print("--- %s seconds ---" % (time.time() - start_time))
		if len(vect_clustered) < num_trocar+2:

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

			plot_cfs_matrix(y_true,y_pred,list(dict_gt.keys()))


		else:
			print(vect_clustered)
			print("Wrongly classify")
			return

		ite = 0

		for key,value in dict_cluster.items():

			if ite == num_trocar:

				break

			vect_start_clustered = np.zeros((len(value),3),dtype=np.float32)
			vect_end_clustered = np.zeros((len(value),3),dtype=np.float32)	

			vect_start_clustered = vect_start_with_noise[value]
			vect_end_clustered = vect_end_with_noise[value]

			vect_rand_clustered = (vect_end_clustered - vect_start_clustered)/(SCALE_COEF1*(SCALE_COEF2+1))

			a,b = generate_coef(vect_rand_clustered, vect_end_clustered)

			final_sol,residuals_err = linear_least_squares(a,b,residuals=True)

			rela_err = relative_err_calc(final_sol,trocar[ite])

			abs_err = abs_err_calc(final_sol,trocar[ite])

			eu_err = eudist_err_calc(final_sol,trocar[ite])
			
			# covar = np.matrix(np.dot(a.T, a)).I
			covar = np.linalg.pinv(np.dot(a.T, a))
			var_mtrx = np.dot(residuals_err,covar)/(a.shape[0]-3+1)
			diagonal = np.diagonal(var_mtrx)
			std_err = np.linalg.norm(diagonal)
			u,s,vh = np.linalg.svd(var_mtrx, full_matrices=True)
			print("Estimated trocar (mm): ",final_sol)
			print("{} ground truth (mm): {}".format(key,trocar[ite]))
			print("Singular values: {} - {} - {}".format(s[0],s[1],s[2]))
			print("Standard error (mm): ",std_err)
			# print("Estimated trocar (mm): ",final_sol)
			# DrawConfidenceRegion(s,final_sol,vh)
			print("Relative error for X,Y,Z respectively (%): {} - {} - {}".format(rela_err[0],rela_err[1],rela_err[2]))
			print("Absolute error for X,Y,Z respectively (mm): {} - {} - {}".format(abs_err[0],abs_err[1],abs_err[2]))
			print("Root mean square error (mm): ",eu_err)
			# DrawConfidenceRegion(s,final_sol,vh,key)	
			list_rela_err[counter,ite,:] = rela_err
			list_abs_err[counter,ite,:] = abs_err
			list_eu_err[counter,ite] = eu_err
			list_std_err[counter,ite] = std_err
			ite += 1

		counter += 1

	#plot the result
	plt.figure(100),
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	fig.suptitle('Relative error comparison')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,0], 'b-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,0], 'g-')
	axs[0,0].plot(list_noise_percentage, list_rela_err[:,3,0], 'm-')
	axs[0,0].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	axs[0,0].set(xlabel='Incorrect data (%)', ylabel='Relative error for X (%)')
	# axs[0,0].set_title('Trocar 1')
	axs[0,1].plot(list_noise_percentage, list_rela_err[:,0,1], 'r-')
	axs[0,1].plot(list_noise_percentage, list_rela_err[:,1,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_rela_err[:,2,1], 'g-')
	axs[0,1].plot(list_noise_percentage, list_rela_err[:,3,1], 'm-')
	axs[0,1].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	axs[0,1].set(xlabel='Incorrect data (%)', ylabel='Relative error for Y (%)')

	axs[1,0].plot(list_noise_percentage, list_rela_err[:,0,2], 'r-')
	axs[1,0].plot(list_noise_percentage, list_rela_err[:,1,2], 'b-')
	axs[1,0].plot(list_noise_percentage, list_rela_err[:,2,2], 'g-')
	axs[1,0].plot(list_noise_percentage, list_rela_err[:,3,2], 'm-')
	axs[1,0].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	axs[1,0].set(xlabel='Incorrect data (%)', ylabel='Relative error for Z (%)')

	plt.figure(200),
	fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	fig.suptitle('Absolute error comparison')
	axs[0,0].plot(list_noise_percentage, list_abs_err[:,0,0], 'r-')
	axs[0,0].plot(list_noise_percentage, list_abs_err[:,1,0], 'b-')
	axs[0,0].plot(list_noise_percentage, list_abs_err[:,2,0], 'g-')
	axs[0,0].plot(list_noise_percentage, list_abs_err[:,3,0], 'm-')
	axs[0,0].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	axs[0,0].set(xlabel='Incorrect data (%)', ylabel='Absolute error for X (mm)')
	# axs[0,0].set_title('Trocar 1')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,1], 'r-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,1], 'b-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,1], 'g-')
	axs[0,1].plot(list_noise_percentage, list_abs_err[:,3,1], 'm-')
	axs[0,1].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	axs[0,1].set(xlabel='Incorrect data (%)', ylabel='Absolute error for Y (mm)')

	axs[1,0].plot(list_noise_percentage, list_abs_err[:,0,2], 'r-')
	axs[1,0].plot(list_noise_percentage, list_abs_err[:,1,2], 'b-')
	axs[1,0].plot(list_noise_percentage, list_abs_err[:,2,2], 'g-')
	axs[1,0].plot(list_noise_percentage, list_abs_err[:,3,2], 'm-')
	axs[1,0].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	axs[1,0].set(xlabel='Incorrect data (%)', ylabel='Absolute error for Z (mm)')

	plt.figure(300),
	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	fig.suptitle('RMSE comparison')
	axs.plot(list_noise_percentage, list_eu_err[:,0], 'r-')
	axs.plot(list_noise_percentage, list_eu_err[:,1], 'b-')
	axs.plot(list_noise_percentage, list_eu_err[:,2], 'g-')
	axs.plot(list_noise_percentage, list_eu_err[:,3], 'm-')
	axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	axs.set(xlabel='Incorrect data (%)', ylabel='RMSE (mm)')

	plt.figure(400),
	fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	fig.suptitle('Standard error comparison')
	axs.plot(list_noise_percentage, list_std_err[:,0], 'r-')
	axs.plot(list_noise_percentage, list_std_err[:,1], 'b-')
	axs.plot(list_noise_percentage, list_std_err[:,2], 'g-')
	axs.plot(list_noise_percentage, list_std_err[:,3], 'm-')
	axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	axs.set(xlabel='Incorrect data (%)', ylabel='Standard error (mm)')

	plt.show()	

	# plydata = PlyData.read("liver_simplified.ply")
	# vertex_data = plydata['vertex'].data # numpy array with fields ['x', 'y', 'z']
	# pts = np.zeros([vertex_data.size, 3])
	# pts[:, 0] = vertex_data['x']
	# pts[:, 1] = vertex_data['y']
	# pts[:, 2] = vertex_data['z']

	visualize_model(trocar=trocar,pts=pts,vect_end=vect_end,vect_start=vect_start,line_idx=dict_gt,gt=True)
		
		# visualize_model(trocar=trocar,vect_end=vect_end,vect_start=vect_start,line_idx=dict_cluster,gt=False)
		# visualize_model(pts=pts)


###################################################################

if __name__ == '__main__':

	trocar = np.array([[-60,70,120],[-65,-70,120],[65,-75,120],[60,75,120]])

	# trocar = np.array([[30,68,125],[150,70,130],[35, 200,120]])
	# run_algo6(trocar,percentage)

	# percentage = np.array([0.35,0.27,0.18,0.2])
	# run_algo4(trocar,percentage)
	# run_algo5(trocar,percentage)
	percentage = np.array([0.4,0.3,0.2,0.1,0.1])
	ransac_new(trocar,percentage)

	# plydata = PlyData.read("liver_simplified.ply")
	# vertex_data = plydata['vertex'].data # numpy array with fields ['x', 'y', 'z']
	# pts = np.zeros([vertex_data.size, 3])
	# pts[:, 0] = vertex_data['x']
	# pts[:, 1] = vertex_data['y']
	# pts[:, 2] = vertex_data['z']
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.set_xlabel('X (mm)')
	# ax.set_ylabel('Y (mm)')
	# ax.set_zlabel('Z (mm)')
	# for i in range(pts.shape[0]):
	# 	ax.scatter(pts[i,0],pts[i,1],pts[i,2],marker = ",",color="#948e8e")
	# plt.show()

	# list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
	# num_trocar = trocar.shape[0]
	# list_rela_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	# list_abs_err = np.zeros((list_noise_percentage.shape[0],num_trocar,3),dtype=np.float32)
	# list_eu_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)
	# list_std_err = np.zeros((list_noise_percentage.shape[0],num_trocar,1),dtype=np.float32)
	# count = 0
	# for num in list_noise_percentage:

	# 	percentage = np.array([0.4,0.3,0.2,0.1,num/100])
	# 	list_rela, list_abs, list_eu, list_std = ransac_new(trocar,percentage)
	# 	for i in range(num_trocar):
	# 		list_rela_err[count,i,:] = list_rela[i,:]
	# 		list_abs_err[count,i,:] = list_abs[i,:]
	# 		list_eu_err[count,i] = list_eu[i]
	# 		list_std_err[count,i] = list_std[i]

	# 	count+=1

	# #plot the result
	# plt.figure(100),
	# fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	# fig.suptitle('Relative error comparison')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,0,0], 'r-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,1,0], 'b-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,2,0], 'g-')
	# axs[0,0].plot(list_noise_percentage, list_rela_err[:,3,0], 'm-')
	# axs[0,0].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# axs[0,0].set(xlabel='Incorrect Data (%)', ylabel='Relative error for X (%)')
	# # axs[0,0].set_title('Trocar 1')
	# axs[0,1].plot(list_noise_percentage, list_rela_err[:,0,1], 'r-')
	# axs[0,1].plot(list_noise_percentage, list_rela_err[:,1,1], 'b-')
	# axs[0,1].plot(list_noise_percentage, list_rela_err[:,2,1], 'g-')
	# axs[0,1].plot(list_noise_percentage, list_rela_err[:,3,1], 'm-')
	# axs[0,1].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# axs[0,1].set(xlabel='Incorrect Data (%)', ylabel='Relative error for Y (%)')

	# axs[1,0].plot(list_noise_percentage, list_rela_err[:,0,2], 'r-')
	# axs[1,0].plot(list_noise_percentage, list_rela_err[:,1,2], 'b-')
	# axs[1,0].plot(list_noise_percentage, list_rela_err[:,2,2], 'g-')
	# axs[1,0].plot(list_noise_percentage, list_rela_err[:,3,2], 'm-')
	# axs[1,0].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# axs[1,0].set(xlabel='Incorrect Data (%)', ylabel='Relative error for Z (%)')

	# plt.figure(200),
	# fig, axs = plt.subplots(2, 2, figsize = (10, 4))
	# fig.suptitle('Absolute error comparison')
	# axs[0,0].plot(list_noise_percentage, list_abs_err[:,0,0], 'r-')
	# axs[0,0].plot(list_noise_percentage, list_abs_err[:,1,0], 'b-')
	# axs[0,0].plot(list_noise_percentage, list_abs_err[:,2,0], 'g-')
	# axs[0,0].plot(list_noise_percentage, list_abs_err[:,3,0], 'm-')
	# axs[0,0].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# axs[0,0].set(xlabel='Incorrect Data (%)', ylabel='Absolute error for X (mm)')
	# # axs[0,0].set_title('Trocar 1')
	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,0,1], 'r-')
	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,1,1], 'b-')
	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,2,1], 'g-')
	# axs[0,1].plot(list_noise_percentage, list_abs_err[:,3,1], 'm-')
	# axs[0,1].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# axs[0,1].set(xlabel='Incorrect Data (%)', ylabel='Absolute error for Y (mm)')

	# axs[1,0].plot(list_noise_percentage, list_abs_err[:,0,2], 'r-')
	# axs[1,0].plot(list_noise_percentage, list_abs_err[:,1,2], 'b-')
	# axs[1,0].plot(list_noise_percentage, list_abs_err[:,2,2], 'g-')
	# axs[1,0].plot(list_noise_percentage, list_abs_err[:,3,2], 'm-')
	# axs[1,0].legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# axs[1,0].set(xlabel='Incorrect Data (%)', ylabel='Absolute error for Z (mm)')

	# plt.figure(300),
	# fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# fig.suptitle('RMSE comparison')
	# axs.plot(list_noise_percentage, list_eu_err[:,0], 'r-')
	# axs.plot(list_noise_percentage, list_eu_err[:,1], 'b-')
	# axs.plot(list_noise_percentage, list_eu_err[:,2], 'g-')
	# axs.plot(list_noise_percentage, list_eu_err[:,3], 'm-')
	# axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# axs.set(xlabel='Incorrect Data (%)', ylabel='RMSE (mm)')

	# plt.figure(400),
	# fig, axs = plt.subplots(1, 1, figsize = (10, 4))
	# fig.suptitle('Standard error comparison')
	# axs.plot(list_noise_percentage, list_std_err[:,0], 'r-')
	# axs.plot(list_noise_percentage, list_std_err[:,1], 'b-')
	# axs.plot(list_noise_percentage, list_std_err[:,2], 'g-')
	# axs.plot(list_noise_percentage, list_std_err[:,3], 'm-')
	# axs.legend(['Trocar 1','Trocar 2','Trocar 3','Trocar 4'])
	# axs.set(xlabel='Incorrect Data (%)', ylabel='Standard error (mm)')

	# plt.show()	