import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.linalg.blas import dgemm

num_data = 1000

liver = np.array([0,30,-25])
std_noise = 5

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
    y1 = w * np.sin(t)
    
    return [x0+x1, y0 +y1, z0]

def generate_correct_data(N_lines, trocar, scale1 = 1, scale2 = 1, sigma = 5):

    """
    Generate simulation data (data with noise follows Gaussian distribution)
    """

    # Generate direction unit vector for all lines passing through trocar

    #solution1
    vect_rand = np.random.normal(0,1,(N_lines,3))
    norm_rand = np.linalg.norm(vect_rand,axis=1)        
    #normalize to get unit vector
    vect_rand = vect_rand/norm_rand[:,np.newaxis]

    # #solution2    
    # vect_rand = np.zeros((N_lines,3),dtype=np.float32)

    # for i in range(N_lines):

    #     vect_rand[i] = random_unit_vector()
    
    #Get direction vectors to offset the lines
    u = np.zeros((N_lines,3),dtype=np.float32)    

    for i in range(N_lines):
        u[i] = np.array([vect_rand[i,1], -vect_rand[i,0], 0])


    # coef = np.random.normal(0,100,(N_lines,2))

    # for i in range(u.shape[0]):
    #     u[i] = coef[i,0]*u1 + coef[i,1]*u2
    #     u[i] = u[i]/np.linalg.norm(u[i])

    # Noisy data is defined based on the distance from the trocar. d ~ N(0,std_noise)
    d = np.random.normal(0,sigma,N_lines)

    # Generate points satify list distances d
    pts = np.zeros((N_lines,3),dtype=np.float32)    
    
    for i in range(pts.shape[0]):
        pts[i] = d[i]*u[i] + trocar

    # unit_vect_stack = np.tile(unit_vect,[N_lines,1]).astype(np.float32)

    vect_start = pts - scale1*scale2*vect_rand
    vect_end = pts + scale1*vect_rand

    return vect_end, vect_start, trocar, vect_rand

def generate_incorrect_data(N_lines, trocar, scale1 = 1, scale2 = 1, sigma = 5, upper_bound = 150):

    """
    Generate incorrect data (uniform distribution)
    """

    # Generate direction unit vector for all lines passing through trocar

    #solution1
    vect_rand = np.random.normal(0,1,(N_lines,3))
    norm_rand = np.linalg.norm(vect_rand,axis=1)        
    #normalize to get unit vector
    vect_rand = vect_rand/norm_rand[:,np.newaxis]

    # #solution2    
    # vect_rand = np.zeros((N_lines,3),dtype=np.float32)

    # for i in range(N_lines):

    #     vect_rand[i] = random_unit_vector()
    
    #Get direction vectors to offset the lines
    u = np.zeros((N_lines,3),dtype=np.float32)    

    for i in range(N_lines):
        u[i] = np.array([vect_rand[i,1], -vect_rand[i,0], 0])

    # Incorrect data is defined based on the distance from the trocar. d ~ U(sigma,upperbound - depends on the scene)
    d = np.random.uniform(1.96*sigma,upper_bound,N_lines)

    # Generate points satify list distances d
    pts = np.zeros((N_lines,3),dtype=np.float32)    
    
    for i in range(pts.shape[0]):
        pts[i] = d[i]*u[i] + trocar

    # unit_vect_stack = np.tile(unit_vect,[N_lines,1]).astype(np.float32)

    vect_start = pts - scale1*scale2*vect_rand
    vect_end = pts + scale1*vect_rand

    return vect_end, vect_start, trocar, vect_rand


def generate_data(N_lines, percentage, trocar, scale1, scale2, sigma = 5, upper_bound = 150):

    num_trocar = trocar.shape[0]


    # Generate lines to each trocar

    vect_end = np.empty((0,3),dtype=np.float32) 
    vect_start = np.empty((0,3),dtype=np.float32)   
    list_idx_gt = []
    cur = 0
    last = 0
    dict_gt = {}
    dict_cluster = {}

    for i in range(num_trocar):

        end_temp, start_temp,_,_ = generate_correct_data(int(N_lines*percentage[i]*(1-percentage[-1])), trocar[i], scale1 = scale1, scale2 = scale2, sigma = sigma) 
        vect_end = np.append(vect_end,end_temp,axis=0)
        vect_start = np.append(vect_start,start_temp,axis=0)
        cur += int(N_lines*percentage[i]*(1-percentage[-1]))
        # print("cur: {}, last: {}",cur,last)

        list_temp = np.arange(last, cur)

        # print("List temp: ",list_temp)

        list_idx_gt.append(list_temp)
        
        last += int(N_lines*percentage[i]*(1-percentage[-1]))

    ### add Incorrect data
    if percentage[-1]:

        for i in range(num_trocar):

            outlier_end, outlier_start,_,_ = generate_incorrect_data(int(N_lines*percentage[i]*percentage[-1]), trocar[i], scale1 = scale1, scale2 = scale2,sigma = sigma,upper_bound = upper_bound)
            vect_end = np.append(vect_end,outlier_end,axis=0)
            vect_start = np.append(vect_start,outlier_start,axis=0)
            cur += int(N_lines*percentage[i]*percentage[-1])
            # print("cur: {}, last: {}",cur,last)

            list_temp = np.arange(last, cur)

            # print("List temp: ",list_temp)

            if i == 0:

                list_idx_gt.append(list_temp.tolist())

            else:

                list_idx_gt[num_trocar].extend(list_temp.tolist())

            last += int(N_lines*percentage[i]*percentage[-1])


    for i in range(len(list_idx_gt)):

        if i == num_trocar:

            dict_gt["Incorrect Data"] = list_idx_gt[i]
        
        else:

            dict_gt["Trocar "+str(i+1)] = list_idx_gt[i]

    return vect_start, vect_end, dict_gt


def generate_perfect_data(N_lines, trocar,scale1 = 1, scale2 = 1):

    # vect_rand = np.random.randint(100, size=(N_lines,3)).astype(np.float32)
    vect_rand = np.zeros((N_lines,3),dtype=np.float32)
    
    for i in range(N_lines):

        vect_rand[i] = random_unit_vector()

    vect_trocar = np.tile(trocar,[N_lines,1]).astype(np.float32)

    vect_start = vect_trocar - scale1*scale2*vect_rand
    vect_end = vect_trocar + scale1*vect_rand

    return vect_end, vect_start, trocar, vect_rand

def generate_outliers(N_outliers, trocar,scale1 = 1, scale2 = 1):
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

    distance = np.arange(10,50,5)
    cydist = cycle(distance)
    for i in range(N_outliers):
        dist = next(cydist)
        outlier = create_random_point(trocar[0], trocar[1], trocar[2], dist)
        outliers.append(outlier)


    outliers = np.array(outliers).astype(np.float32)

    # Generate outlier lines
    # vect_outlier_rand = np.random.randint(500, size=(N_outliers,3)).astype(np.float32)

    vect_outlier_rand = np.zeros((N_outliers,3),dtype=np.float32)

    for i in range(N_outliers):
        vect_outlier_rand[i] = random_unit_vector()
    
    vect_outlier_start = outliers - scale1*scale2*vect_outlier_rand
    vect_outlier_end = scale1*vect_outlier_rand + outliers

    return vect_outlier_end, vect_outlier_start, vect_outlier_rand

def add_gaussian_noise(data1,data2, mean=0, sigma=0.1, percentage = 0.2):

    if sigma == 0:
        
        return data1,data2

    num_data = len(data1)

    num_outlier = int(num_data*percentage)

    gaussian = np.random.normal(mean,sigma,(num_outlier,3))

    random_list = np.random.randint(0,num_data,size=num_outlier)

    j = 0

    data_with_noise1 = data1
    data_with_noise2 = data2

    for i in random_list:

        data_with_noise1[i] = data_with_noise1[i] + gaussian[j]
        data_with_noise2[i] = data_with_noise2[i] + gaussian[j]
        j += 1
    
    return data_with_noise1,data_with_noise2,random_list   

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

    """
    Parameters
    unit_vect : np.ndarray of shape (n,3)
    point : np.ndarray of shape (n,3)

    Return
    coefficient of ||aX-b||
    a : np.ndarray of shape (n*3,3)
    b : np.ndarray of shape (n*3,1)
    """
    I = np.eye(3,dtype=np.float32)
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

