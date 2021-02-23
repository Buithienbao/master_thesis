import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


# Trocar coordinates fixed values

trocar_c = [30,68,102]
trocar_c = np.array(trocar_c).astype(np.float32)

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

def generate_perfect_data(N_lines = 100):

    vect_rand = np.random.randint(100, size=(N_lines,3)).astype(np.float32)

    vect_trocar = np.tile(trocar_c,[N_lines,1]).astype(np.float32)
    vect_end = vect_trocar + vect_rand
    vect_start = vect_trocar - vect_rand

    return vect_end, vect_start, trocar_c

def generate_outliers(N_outliers = 20):

    # Generate outlier points

    outliers = []

    for i in range(N_outliers):

      outlier = create_random_point(trocar_c[0], trocar_c[1], trocar_c[2], 15)
      outliers.append(outlier)


    outliers = np.array(outliers).astype(np.float32)

    # Generate outlier lines
    vect_outlier_rand = np.random.randint(100, size=(N_outliers,3)).astype(np.float32)

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

    for i in random_list:

        data[i] = data[i] + gaussian[j]

        j += 1

    return data    

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


