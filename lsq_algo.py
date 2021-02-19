import numpy as np 
import shapely
from generate_data import generate_perfect_data,generate_outliers
from scipy.optimize import least_squares

def lineseg_dist(p, a, b):

	dist = 0

	for i in range(len(a)):

		x = a[i] - b[i]
		t = np.dot(p-b[i], x)/np.dot(x, x)
		dist += np.linalg.norm(t*(a[i]-b[i])+b[i]-p)
	
	return dist


a_train,b_train,gt = generate_perfect_data()

outlier_a, outlier_b = generate_outliers()

c_train = np.concatenate((a_train,outlier_a))
# print(c_train.shape)

p0 = np.array([50,50,50]).astype(np.float32)

res_soft_l1 = least_squares(lineseg_dist, p0, loss='soft_l1', f_scale=0.1,

                            args=(a_train, b_train))


res_soft_l2 = least_squares(lineseg_dist, p0, loss='linear', f_scale=0.1,

                            args=(a_train, b_train))

print(res_soft_l1.x)

print(res_soft_l2.x)
# p = np.array([1,2,3])
# a = np.array([2,4,5])
# b = np.array([3,5,7])

# print(lineseg_dist(p,a,b))

# def dist_func()