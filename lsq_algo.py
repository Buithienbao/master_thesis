import numpy as np 
import shapely
from generate_data import generate_perfect_data,generate_outliers,add_gaussian_noise
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

LOOP_NUM = 10
N_lines = 1000
# percentage = 0.2
# num_outliers = int(N_lines*percentage)

start_range = 0
end_range = 50
step = (end_range - start_range)/10


def lineseg_dist(p, a, b):

	dist = 0

	for i in range(len(a)):

		x = a[i] - b[i]
		t = np.dot(p-b[i], x)/np.dot(x, x)
		dist += np.linalg.norm(t*(a[i]-b[i])+b[i]-p)
	
	return dist

def relative_err_calc(pred,gt):

	return np.abs(pred-gt)/gt*100

def abs_err_calc(pred,gt):

	return np.abs(pred-gt)

def eudist_err_calc(pred,gt):

	return np.linalg.norm(pred-gt)



p0 = np.array([50,50,50]).astype(np.float32)

a_train,b_train,gt = generate_perfect_data(N_lines = N_lines)

list_noise_percentage = np.arange(start_range, end_range + step,step,dtype=np.uint8)
# print(list_noise)
list_rela_err_L1 = []

list_abs_err_L1 = []

list_rela_err_L2 = []

list_abs_err_L2 = []

list_eu_err_L1 = []

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

		# outlier_a, outlier_b = generate_outliers(N_outliers = num)

		# c_train = np.concatenate((a_train,outlier_a))

		# d_train = np.concatenate((b_train,outlier_b))




		for i in range(LOOP_NUM):

			a_train_with_noise = add_gaussian_noise(a_train,mean=0,var=num,percentage=0.2)
			# a_train_with_noise = add_gaussian_noise(a_train,mean=0,var=10,percentage=num/100)

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
axs[0,0].set(xlabel='Sigma Value', ylabel='Relative error (%)')

axs[0,1].plot(list_noise_percentage, list_rela_err_L2[:,0], 'r-')
axs[0,1].plot(list_noise_percentage, list_rela_err_L2[:,1], 'b-')
axs[0,1].plot(list_noise_percentage, list_rela_err_L2[:,2], 'g-')
axs[0,1].legend(['L2-norm -- Relative error for X','L2-norm -- Relative error for Y','L2-norm -- Relative error for Z'])
axs[0,1].set(xlabel='Sigma Value', ylabel='Relative error (%)')

axs[1,0].plot(list_noise_percentage, list_abs_err_L1[:,0], 'r-')
axs[1,0].plot(list_noise_percentage, list_abs_err_L1[:,1], 'b-')
axs[1,0].plot(list_noise_percentage, list_abs_err_L1[:,2], 'g-')
axs[1,0].legend(['L1-norm -- Absolute error for X','L1-norm -- Absolute error for Y','L1-norm -- Absolute error for Z'])
axs[1,0].set(xlabel='Sigma Value', ylabel='Absolute error')

axs[1,1].plot(list_noise_percentage, list_abs_err_L2[:,0], 'r-')
axs[1,1].plot(list_noise_percentage, list_abs_err_L2[:,1], 'b-')
axs[1,1].plot(list_noise_percentage, list_abs_err_L2[:,2], 'g-')
axs[1,1].legend(['L2-norm -- Absolute error for X','L2-norm -- Absolute error for Y','L2-norm -- Absolute error for Z'])
axs[1,1].set(xlabel='Sigma Value', ylabel='Absolute error')


axs[2,0].plot(list_noise_percentage, list_eu_err_L1, 'r--')
axs[2,0].plot(list_noise_percentage, list_eu_err_L2, 'b-')
axs[2,0].legend(['L1-norm -- Euclidean dist', 'L2-norm -- Euclidean dist'])
axs[2,0].set(xlabel='Sigma Value', ylabel='Euclidean dist')

plt.show()