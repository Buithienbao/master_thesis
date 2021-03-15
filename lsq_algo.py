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

	dist = []
	dist_val = 0
	for i in range(len(a)):

		x = a[i] - b[i]
		t = np.dot(p-b[i], x)/np.dot(x, x)
		dist_val = np.linalg.norm(t*(a[i]-b[i])+b[i]-p)
		dist.append(dist_val)
	
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

	print('list_rela_err_L1: ',list_rela_err_L1)
	print('list_rela_err_L2: ',list_rela_err_L2)

	print('list_abs_err_L1: ',list_abs_err_L1)
	print('list_abs_err_L2: ',list_abs_err_L2)

	print('list_eu_err_L1: ',list_eu_err_L1)
	print('list_eu_err_L2: ',list_eu_err_L2)

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

	# a_train,b_train,gt,vect = generate_perfect_data(N_lines = N_lines)

	a,b = random_coef()
	




###################################################################



if __name__ == '__main__':
	run_algo1()
	random_unit_vector()