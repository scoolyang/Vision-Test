import cv2
import os
import numpy as np
import statistics
import matplotlib.pyplot as plt; plt.ion()
from fastai.imports import *

class ReadFileInfo(object):              ## A class used to read npy file coming from the output of NN.
    def __init__(self, file_name):
        self._fileinfo = np.load(file_name)
        self._all_image_name_index = self.show_all_image_name_index()

    def show_all_image_name_index(self):           ## Generate a list with all index and image name
        coor_image_index_list = []
        for i in range(len(self._fileinfo)):
            if self._fileinfo[i][0][0] not in coor_image_index_list:
                coor_image_index_list.append([i, self._fileinfo[i][0][0]])
        return coor_image_index_list

    def show_all_image_name(self):                  ## Generate a list with all image name and show the list as output
        image_name_list = []
        for i in range(len(self._all_image_name_index)):
            image_name_list.append(self._all_image_name_index[i][1])
        print(image_name_list)
        return image_name_list

    def number_of_obects(self, image_name):         ## show number of objects in a given image
        for i in range(len(self._all_image_name_index)):
            if image_name == self._all_image_name_index[i][1]:
                return len(self._fileinfo[i])
        return('There is no file that names ' + image_name)

    def BS_image_index(self, image_name):          ## show the corrosponding index of a given image
        for i in range(len(self._all_image_name_index)):
            if image_name == self._all_image_name_index[i][1]:
                return self._all_image_name_index[i][0]
        return('There is no file that names ' + image_name)

    def show_bounding_box(self, image_name):      ## show the bounding of a given image and return 4 values used to generate bounding box
        img = cv2.imread(image_name, 0)
        image_index = self.BS_image_index(image_name)
        for i in range(self.number_of_obects(image_name)):
            xmin = int(self._fileinfo[image_index][i][1])
            ymin = int(self._fileinfo[image_index][i][2])
            xmax = xmin + int(self._fileinfo[image_index][i][3])
            ymax = ymin + int(self._fileinfo[image_index][i][4])
            if img is None:
                break
            roi_img = img[ymin: ymax, xmin: xmax]
            cv2.imshow("Image", roi_img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        return ymin, ymax, xmin, xmax

class Three_D_Position(object):              ## used to transfer pixel coordinate to world coordinate output is a 1*4 matrix
    def __init__(self, pixel_coor, intrinsic_mat, extrinsic_rotation, extrinsic_translation, camera_location):
        self._u = pixel_coor[0]
        self._v = pixel_coor[1]
        self._depth = pixel_coor[2]
        self._intrinsic_mat = intrinsic_mat
        self._ex_rot_mat = extrinsic_rotation
        self._ex_tran_mat = extrinsic_translation
        self._camera_location = camera_location

    def pixel_to_optical(self):
        pixel_coor_depth_frame = np.array([self._u, self._v, 1])
        temp_mat = (self.get_inverse_matrix(self._intrinsic_mat) @ pixel_coor_depth_frame) * self._depth
        optical_z = temp_mat[2]
        optical_x = temp_mat[0]
        optical_y = temp_mat[1]
        optical_coor = np.array([optical_x, optical_y, optical_z])
        return optical_coor

    def optical_to_world(self):
        Final_mat = np.eye(4)
        pos_wc_mat_x = self._camera_location[0]
        pos_wc_mat_y = self._camera_location[1]
        pos_wc_mat_z = self._camera_location[2] # unit meters
        pos_wc_mat = np.array([pos_wc_mat_x, pos_wc_mat_y, pos_wc_mat_z])

        R_wc_mat = self._ex_rot_mat
        R_oc_mat = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        temp1 = np.matmul(R_oc_mat, np.transpose(R_wc_mat))
        temp2 = np.matmul(-1 * R_oc_mat, np.transpose(R_wc_mat))
        temp3 = np.matmul(temp2, pos_wc_mat).reshape(3, 1)

        Final_mat[0:3, 0:3] = temp1
        Final_mat[0, 3] = temp3[0]
        Final_mat[1, 3] = temp3[1]
        Final_mat[2, 3] = temp3[2]

        optical_coor = np.concatenate((self.pixel_to_optical().reshape(3, 1), [[1]]), axis=0)
        world_coor = np.matmul(self.get_inverse_matrix(Final_mat), optical_coor)

        return world_coor

    def get_inverse_matrix(self, matrix):
        return np.linalg.inv(matrix)

def latlngtoGlobalxy(lat, lng, alt):         ## Used to tranfer latitude and longtitude value to a cartisian coordinate
    lat_deg_2_rad = lat * np.pi / 180
    lon_deg_2_rad = lng * np.pi / 180
    a = 6378137
    e = 8.1819190842622 * 10 ** -2
    N = a / np.sqrt(1 - e ** 2 * (np.sin(lat_deg_2_rad) ** 2))
    x_pos = (N + alt) * np.cos(lat_deg_2_rad) * np.cos(lon_deg_2_rad)
    y_pos = (N + alt) * np.cos(lat_deg_2_rad) * np.sin(lon_deg_2_rad)
    z_pos = ((1 - e ** 2) * N + alt) * np.sin(lat_deg_2_rad)

    final_pos = [x_pos, y_pos, z_pos]

    return final_pos

def transform_frame(pitch, yaw, roll):    ## Used to tranform orientational direction
    roll = roll * np.pi / 180
    yaw = yaw * np.pi / 180
    pitch = pitch * np.pi / 180
    pitch_rotation_matrix = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                                    [0, 1, 0],
                                    [-np.sin(pitch), 0, np.cos(pitch)]])
    roll_rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(roll), -np.sin(roll)],
                                    [0, np.sin(roll), np.cos(roll)]])
    yaw_rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0],
                                    [0, 0, 1]])
    transform_matrix = np.eye(3)
    transform_matrix = pitch_rotation_matrix * yaw_rotation_matrix * roll_rotation_matrix

    return transform_matrix

def Generate_Object_Depth_Pair(number_of_objects, ymin, ymax, xmin, xmax, depth_map): ## Used to generate object depth information pair
    object_depth_median_pair_dep = []
    for k in range(number_of_objects):
        roi_img_dep = abs(depth_map[ymin: ymax, xmin: xmax])
        roi_center_v = round(roi_img_dep.shape[0] / 2, 0) + ymin
        roi_center_u = round(roi_img_dep.shape[1] / 2, 0) + xmin
        pixel_depth_value_list_dep = []
        for v in range(roi_img_dep.shape[0]):
            for u in range(roi_img_dep.shape[1]):
                dist = roi_img_dep[v, u][0]
                if dist > 0:
                    pixel_depth_value_list_dep.append(dist)
        if pixel_depth_value_list_dep == []:
            pixel_depth_value_list_dep.append(0)
        median_dep = statistics.median(pixel_depth_value_list_dep)
        median_pair_dep = [roi_center_u, roi_center_v, median_dep]
        object_depth_median_pair_dep.append(median_pair_dep)

    return object_depth_median_pair_dep

if __name__ == '__main__':
    CAMERA = 'far'           ## can changed to far or near, also if there are three cameras, this value can be set into 3 or more values.
    lat_far = 32.888665
    lon_far = -117.241121
    lat_near = 32.888958
    lon_near = -117.240921
    File_Info_Name = 'street_data_left1a.npy' ## given npy files including bounding boxes information
##    cor_depth_map = 'DEPTH_MAP_NAME'  ## given depth images corrosponding to the images in npy files



    if CAMERA == 'far':
        camera_type = 'far'

        intrinsic_mat = np.array([[905.973, 0, 626.31],
                                  [0, 905.973, 346.674],
                                  [0, 0, 1]])

        extrinsic_rot_mat = np.array([[-0.8660254, -0.5, 0],
                                      [0.5, -0.8660254, 0],
                                      [0, 0, 1]])

        extrinsic_tran_mat = [0.015, 0, 0]

        far_camera_pos = latlngtoGlobalxy(lat_far, lon_far, 0)
        near_camera_pos = latlngtoGlobalxy(lat_near, lon_near, 0)
        far_camera_pos_abs = [far_camera_pos[0] - near_camera_pos[0], far_camera_pos[1] - near_camera_pos[1], 0]
        transform_matrix = transform_frame(0, 180, 0)  ## this 180 degree depends on the information coming from the test. It can be changed to different values.
        camera_location = [0, 0, 0]

    else:
        camera_type = 'near'

        intrinsic_mat = np.array([[895.794, 0, 634.849],
                                  [0, 895.794, 361.893],
                                  [0, 0, 1]])

        extrinsic_rot_mat = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])

        extrinsic_tran_mat = [0.015, 0, 0]

        camera_location = [0, 0, 0]

    world_coor_x_list = []
    world_coor_y_list = []
    world_coor_z_list = []

    File_Info_Object = ReadFileInfo(File_Info_Name)
    Image_Name_List = File_Info_Object.show_all_image_name()

    for i in Image_Name_List:
        Desired_Img_Name = i
        number_of_objects = File_Info_Object.number_of_obects(Desired_Img_Name)
        ymin, ymax, xmin, xmax = File_Info_Object.show_bounding_box(Desired_Img_Name)

        # ####
        # Generate a Depth Map Corrosponding to Desired Img. Should Have the Same Dimension Of the Original Image.
        # cor_depth_map = 'DEPTH_MAP_NAME'
        # ####

        objects_coor_info = Generate_Object_Depth_Pair(number_of_objects, ymin, ymax, xmin, xmax, cor_depth_map)

        for k in range(len(objects_coor_info)):
            my_3D_Projection_flo = Three_D_Position(objects_coor_info[k], intrinsic_mat, extrinsic_rot_mat,
                                                    extrinsic_tran_mat, camera_location)
            cur_world_coor = my_3D_Projection_flo.optical_to_world()
            if CAMERA == 'far':
                cur_world_coor_np = cur_world_coor.reshape(1, 4)
                cur_world_coor = np.matmul(cur_world_coor_np, transform_matrix).reshape(4, 1)
                cur_world_coor[0] = cur_world_coor[0] + abs(far_camera_pos_abs[0])    ##Transfer far camera coodinate to near camera coordiante
                cur_world_coor[1] = cur_world_coor[1] + abs(far_camera_pos_abs[1])    ##Transfer far camera coodinate to near camera coordiante

            world_coor_x_list.append(cur_world_coor[0])
            world_coor_y_list.append(cur_world_coor[1])

        fig = plt.figure()
        print(world_coor_x_list, world_coor_y_list)
        plt.scatter(world_coor_x_list, world_coor_y_list, s=5, alpha=1.0,
                    label='Objects Relative Position from Optical Flow')

        plt.xlabel('Absolute X Position unit(m)')
        plt.ylabel('Absolute Y Position unit(m)')
        plt.title('X and Y Position Value from Depth Image')
        plt.legend(loc=1)
        plt.draw()
        plt.waitforbuttonpress(0)  # this will wait for indefinite time
        plt.show()

