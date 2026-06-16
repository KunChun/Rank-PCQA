import numpy as np
import time
import open3d as o3d
import os
from PIL import Image
import cv2
import argparse


def background_crop(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray_img.shape)
    col = np.mean(gray_img, axis=0)
    row = np.mean(gray_img, axis=1)
    row_a = -1
    row_b = -1
    col_a = -1
    col_b = -1

    for i in range(len(col)):
        if col[i] != 255:
            col_a = i
            break
    for i in range(len(col)):
        if col[-i] != 255:
            col_b = len(col) - i
            break
    for i in range(len(row)):
        if row[i] != 255:
            row_a = i
            break
    for i in range(len(row)):
        if row[-i] != 255:
            row_b = len(row) - i
            break
    # limit the image size
    if (row_b-row_a) < 225:
        flag = row_a + (row_b - row_a)//2
        row_a = flag - 112
        row_b = flag + 113

    if (col_b-col_a) < 225:
        flag = col_a + (col_b - col_a)//2
        col_a = flag - 112
        col_b = flag + 113

    img = img[row_a:row_b, col_a:col_b, :]
    return img


def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


# Camera Rotation
def camera_rotation(type, path, img_path):
    print(path)
    if type == 'ply':
        obj = o3d.io.read_point_cloud(path)
    elif type == 'mesh':
        obj = o3d.io.read_triangle_mesh(path, True)
        obj.compute_vertex_normals()

    if not os.path.exists(img_path + '/'):
        os.mkdir(img_path + '/')

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1080, height=1920)
    vis.add_geometry(obj)
    ctrl = vis.get_view_control()

    interval = 5.82  # interval for 1 degree
    start = time.time()

    # begin rotation rotate the camera on the pathway of (x^2 + y^2 = r^2, z = 0) and (y^2 + z^2 = r^2, x = 0)
    rotate_para = [[0, 0], [90 * interval, 0], [90 * interval, 0], [90 * interval, 0]]
    # rotate_para = [[0, 0], [0, 90 * interval], [0, 90 * interval], [0, 90 * interval]]
    # ctrl.rotate(0, 270 * interval)

    for i in range(4):
        ctrl.rotate(rotate_para[i][0], rotate_para[i][1])
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(True)
        img = Image.fromarray((np.asarray(img) * 255).astype(np.uint8))
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # crop the main object out of the white background
        img = background_crop(img)
        cv2.imwrite(os.path.join(img_path, str(i) + '.png'), img)

    end = time.time()
    print("time consuming: ", end - start)
    vis.destroy_window()
    del ctrl
    del vis


def projection(type, path, img_path):
    # find all the objects
    objs = os.walk(path)
    cnt = 0
    for path, dir_list, file_list in objs:
        cnt = cnt + 1
        for obj in file_list:
            # for textured mesh
            if type == 'mesh':
                # for tmq source
                if obj.endswith('.obj') in path:
                    one_object_path = os.path.join(path, obj)
                    camera_rotation(type, one_object_path, generate_dir(os.path.join(img_path, obj)))
                else:
                    continue
            # for colored point clouds
            elif type == 'ply':
                one_object_path = os.path.join(path, obj)
                camera_rotation(type, one_object_path, generate_dir(os.path.join(img_path, obj)))


if __name__ == '__main__':
    # # capture the projections of the 3d model
    # pc_type = 'ply'
    # ply_path = 'D:/Working/PC_database/PCQA/Referenced_distorted_ply/All_data/distorted_pcs'
    # img_path = 'D:/Working/PC_database/OpenDataset/p3qa_projections'

    # # pc_names = ['barrel', 'basketball', 'bed', 'boat', 'cat', 'chair', 'clock', 'corset',
    # #             'dennis', 'ding_censer', 'dish', 'fish', 'guanyin', 'hayagriva', 'housewife', 'Longdress', 'Loot',
    # #             'mushroom', 'pepper', 'RedAndBlack', 'rooster', 'ryan', 'sewing_machinel', 'shiva', 'side_chair',
    # #             'skateboard', 'skirt', 'Soldier', 'spiderman', 't-pose_man', 'turtle', 'tvset', 'vase', 'violin',
    # #             'walking_man', 'woman']
    # pc_names = ['woman']  # 重新投影 'dish', 'skateboard', 'violin'
    # # pc_names = ['apple', 'axe', 'banana', ]

    # for i in pc_names:
    #     print('='*60+i+'='*60)
    #     pcpath = ply_path + '/' + i + '/'
    #     projection(pc_type, pcpath, img_path)
    
    #=====================================================================================================
    # LS-PCQA dataset: capture the projections of the 3d model
    pc_type = 'ply'
    ply_path = 'D:/Working/PC_database/Opendataset/LS-PCQA/samples_with_MOS/'
    img_path = 'D:/Working/PC_database/OpenDataset/lspcqa_projections'


    # all_files = [f for f in os.listdir(ply_path)]#输出根path下的所有文件名到一个列表中

    # print(all_files)
    
    # for i in all_files:
        # print('='*60+i+'='*60)
        # pcpath = ply_path + '/' + i + '/'
    projection(pc_type, ply_path, img_path)
        