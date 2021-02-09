import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D



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

# Trocar coordinates set to T = [2,6,12]

trocar_c = [30,68,102]

# Number of sample lines

N_lines = 100



vect_rand = np.random.randint(100, size=(N_lines,3))
print(vect_rand)


vect_trocar = np.tile(trocar_c,[100,1])
vect_end = vect_trocar + vect_rand
vect_start = vect_trocar - vect_rand

# Generate outlier points

N_outliers = 20

outliers = []

for i in range(N_outliers):

	outlier = create_random_point(trocar_c[0], trocar_c[1], trocar_c[2], 15)
	outliers.append(outlier)

# Generate outlier lines
vect_outlier_rand = np.random.randint(100, size=(N_outliers,3))

vect_outlier_start = outliers - vect_outlier_rand
vect_outlier_end = vect_outlier_rand + outliers

# Draw 3d graph

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#draw outliers point
# for i in range(N_outliers):
# 	ax.scatter(outliers[i][0],outliers[i][1],outliers[i][2],marker = "o")

#draw outlier lines
for i in range(N_outliers):
    ax.plot([vect_outlier_start[i][0], vect_outlier_end[i][0]], [vect_outlier_start[i][1],vect_outlier_end[i][1]],zs=[vect_outlier_start[i][2],vect_outlier_end[i][2]])


#draw samples lines passing through trocar
for i in range(N_lines):
    ax.plot([vect_start[i][0], vect_end[i][0]], [vect_start[i][1],vect_end[i][1]],zs=[vect_start[i][2],vect_end[i][2]])


#draw trocar point
# ax.scatter(trocar_c[0], trocar_c[1], trocar_c[2], marker = "*")

plt.show()


