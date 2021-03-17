import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from scipy.linalg.blas import dgemm
# Trocar coordinates fixed values

trocar_c = [30,68,102]
trocar_c = np.array(trocar_c).astype(np.float32)

num_data = 100

def get_trocar_gt():
    
    return trocar_c

def create_random_point(x0,y0,z0,distance):
    """
            Utility method for simulation of the points
    """   
    
    r = distance
    u = np.random.uniform(0,1)
    v = np.random.uniform(0,1)
    w = r * np.sqrt(u)
    t = 2 * np.pi * v
    x = w * np.cos(t)
    x1 = x / np.cos(y0)
    y = w * np.sin(t)
    
    return [x0+x1, y0 +y, z0]

def generate_perfect_data(N_lines = num_data, trocar = trocar_c):

    vect_rand = np.random.randint(100, size=(N_lines,3)).astype(np.float32)

    vect_trocar = np.tile(trocar,[N_lines,1]).astype(np.float32)
    vect_end = vect_trocar + vect_rand
    vect_start = vect_trocar - vect_rand

    return vect_end, vect_start, trocar_c, vect_rand

def generate_outliers(N_outliers = 20, trocar = trocar_c):
    """
    generate outlier data

    Parameters
    ----------
    N_outliers : int
        number of outlier data, an integer value

    trocar : numpy.ndarray
        coordinate of trocar center(s), a 2 dimensions array of size (n,3)

    Returns
    -------
    vector_outlier_start : 

    """

    # Generate outlier points

    outliers = []

    for i in range(N_outliers):

      outlier = create_random_point(trocar[0], trocar[1], trocar[2], 15)
      outliers.append(outlier)


    outliers = np.array(outliers).astype(np.float32)

    # Generate outlier lines
    vect_outlier_rand = np.random.randint(500, size=(N_outliers,3)).astype(np.float32)

    vect_outlier_start = outliers - vect_outlier_rand
    vect_outlier_end = vect_outlier_rand + outliers

    return vect_outlier_end, vect_outlier_start

def add_gaussian_noise(data, mean=0, var=0.1, percentage = 0.2):

    num_data = len(data)

    num_outlier = int(num_data*percentage)

    sigma = np.sqrt(var)

    gaussian = np.random.normal(mean,sigma,(num_outlier,3))

    random_list = np.random.randint(0,num_data,size=num_outlier)

    j = 0

    data_with_noise = data

    for i in random_list:

        data_with_noise[i] = data_with_noise[i] + gaussian[j]
  
        j += 1
    
    return data_with_noise    

# # Draw 3d graph

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# #draw outliers point
# # for i in range(N_outliers):
# #     ax.scatter(outliers[i][0],outliers[i][1],outliers[i][2],marker = "o")

# #draw outlier lines
# for i in range(N_outliers):
#     ax.plot([vect_outlier_start[i][0], vect_outlier_end[i][0]], [vect_outlier_start[i][1],vect_outlier_end[i][1]],zs=[vect_outlier_start[i][2],vect_outlier_end[i][2]])


# #draw samples lines passing through trocar
# for i in range(N_lines):
#     ax.plot([vect_start[i][0], vect_end[i][0]], [vect_start[i][1],vect_end[i][1]],zs=[vect_start[i][2],vect_end[i][2]])


# #draw trocar point
# # ax.scatter(trocar_c[0], trocar_c[1], trocar_c[2], marker = "*")

# plt.show()


#############################################################################

def generate_trocars(num = 1):

    trocars = np.random.randint(300, size=(num,3)).astype(np.float32)

    return trocars

def random_unit_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """


    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    vect = (x,y,z)
    vect = np.asarray(vect, dtype=np.float32)
    return vect

def random_point_based_on_unit_vect(unit_vect,trocar):

    N_vect = len(unit_vect)

    vect_trocar = np.tile(trocar,[N_vect,1]).astype(np.float32)

    point = vect_trocar + unit_vect
    # point = point[np.newaxis,:]

    return point

def generate_coef(unit_vect, point):

    I = np.array([[1,0,0],[0,1,0],[0,0,1]])
    num_vect = len(unit_vect)

    a = np.zeros((num_vect*3,3),dtype = np.float32)
    b = np.zeros((num_vect*3,1),dtype = np.float32)

    for i in range(num_vect):

        temp_a = I - np.dot(unit_vect[i].T, unit_vect[i])
        temp_b = np.dot(temp_a,point[i].T)
        
        for j in range(3):

            a[3*i+j] = temp_a[j]
            b[3*i+j] = temp_b[j]

    return a,b

