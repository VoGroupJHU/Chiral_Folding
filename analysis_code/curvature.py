import numpy as np
import math
import gsd.hoomd
import os

# calculate the Menger curvature based on the math definition

size = 1.0
path = "/data/tvo12/xzhan357/project_9_single_chain_chiral/hollow_tube/batch_run/BP_size=" + str(size) + "/"
chain_length = 200
# if size == 0.8:
    # set_folders = [0.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 24.0, 28.0, 30.0, 32.0, 36.0, 40.0, 48.0] # size = 0.8
# if size == 0.9:
    # set_folders = [0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 30.0, 34.0, 38.0, 40.0, 42.0, 44.0, 48.0]
# if size == 1.0:
    # set_folders = [0.1, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 25.0, 30.0, 34.0, 38.0, 42.0, 44.0, 46.0, 48.0]   # size = 1.0    
# if size == 1.1:
    # set_folders = [0.1, 4.0, 8.0, 10.0, 12.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 36.0, 42.0, 48.0, 50.0, 52.0, 54.0, 60.0]

# read all folders in the directory 
folders_dis = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
set_folders = sorted(folders_dis, key=lambda folder_name: float(folder_name.split('=')[-1]))
print(set_folders)

# output file name
name_backup = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen"]
output_file = open(path + "BP_" + str(size) + "_curvature.txt", 'w')
# perform the calculation for all of folders in the directory
for folder in set_folders:
    # read bond length value
    bond_length = folder.split('=')[-1]
    bond_length = f"{float(bond_length):.1f}"    
    folder = folder + "/"    
    # folder = "EQBL=" + (str(int(bond_length)) if bond_length.is_integer() else str(bond_length)) + "/"
    start_name = "_chiral_a_" + str(bond_length)
    filename = start_name
    for name in name_backup:
        test_name = filename + "_" + name
        if os.path.isfile(path + folder + test_name + ".gsd"):
            filename = test_name
        else:
            break
    file_temp = path + folder + filename + '.gsd'
    print("a", file_temp)
    # open the gsd file
    f = gsd.hoomd.open(name = file_temp, mode = 'rb')
    frame_temp = f[len(f)-1]
    f.close()    
    N = frame_temp.particles.N
    pentagon_type = 3   # pentagon type larger than 3!
    backbone_P_type = 0
    # array for backbone particle and pentagon curvature
    pentagon_cu = np.zeros(chain_length-2)
    backbone_cu = np.zeros(chain_length-2)
   # calculate tortosity for backbone and pentagon

    count_1 = 0
    count_2 = 0
    
    # calculate the curvature based on the math definition
    for i in range(N):
        if (i > 11):
            if (frame_temp.particles.typeid[i] == backbone_P_type):
                first_point = frame_temp.particles.position[i-12]
                secon_point = frame_temp.particles.position[i-6]
                third_point = frame_temp.particles.position[i]
                area = 0.5*np.linalg.norm(np.cross(third_point-secon_point, secon_point-first_point))
                norm_1 = np.linalg.norm(secon_point - first_point)
                norm_2 = np.linalg.norm(third_point - first_point)
                norm_3 = np.linalg.norm(third_point - secon_point)
                backbone_cu[count_1] = 4.0*area/norm_1/norm_2/norm_3
                # print(area, norm_1, norm_2, norm_3)
                # print("BP", backbone_cu[count_1])
                count_1 = count_1 + 1
            if (frame_temp.particles.typeid[i] > pentagon_type):
                first_point = frame_temp.particles.position[i-12]
                secon_point = frame_temp.particles.position[i-6]
                third_point = frame_temp.particles.position[i]
                area = 0.5*np.linalg.norm(np.cross(third_point-secon_point, secon_point-first_point))
                norm_1 = np.linalg.norm(secon_point - first_point)
                norm_2 = np.linalg.norm(third_point - first_point)
                norm_3 = np.linalg.norm(third_point - secon_point)
                pentagon_cu[count_2] = 4.0*area/norm_1/norm_2/norm_3
                # print("5gon", pentagon_cu[count_2])
                count_2 = count_2 + 1                
    output_file.write(f"{bond_length} {np.mean(backbone_cu):.5f} {np.std(backbone_cu):.5f} {np.mean(pentagon_cu):.5f} {np.std(pentagon_cu):.5f}\n")
    
output_file.close()                
    