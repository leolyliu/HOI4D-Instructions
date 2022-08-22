import os
import cv2
import argparse
import open3d as o3d
from plyfile import PlyData
import pandas as pd
import scipy.spatial as spt
import numpy as np
import multiprocessing as mlp
from pixel2category import get_mask_and_label


def get_foreground_depths_and_labels(depth_path, mask_path, ds, ls, ls_instanceseg):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    assert mask.shape == (1080, 1920, 3)

    ans = []
    for d in ds:
        x = depth.copy()
        x[~d] = 0
        ans.append(x)
    return ans, ls, ls_instanceseg


def convert(id, depth_path, mask_path, output_background_path):
    # background
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    assert mask.shape == (1080, 1920, 3)

    s = np.sum(mask, axis=2)
    depth_b = depth.copy()
    depth_b[s > 0] = 0
    output_background_p = os.path.join(output_background_path, id + ".png")
    cv2.imwrite(output_background_p, depth_b)
    depth_b = o3d.io.read_image(output_background_p)

    # foreground
    depth_fs = []
    ds, ls, ls_instanceseg = get_mask_and_label(mask_path)
    depths, labels, labels_instanceseg = get_foreground_depths_and_labels(depth_path, mask_path, ds, ls, ls_instanceseg)
    for i in range(len(depths)):
        output_foreground_p = os.path.join(output_background_path.replace("background", "foreground_" + str(i + 1)), id + ".png")
        os.makedirs(os.path.dirname(output_foreground_p), exist_ok=True)
        d = depths[i]
        cv2.imwrite(output_foreground_p, d)
        d_f = o3d.io.read_image(output_foreground_p)
        depth_fs.append([d_f, labels[i], labels_instanceseg[i]])
    
    return depth_fs, depth_b


def scene2frame(filelist, image_path, depth_path, mask_path, output_label_path, output_background_path, extrinsics, label, ckt, max_instance_label):

    os.makedirs(output_label_path, exist_ok=True)
    os.makedirs(output_background_path, exist_ok=True)

    for filename in filelist:
        id = filename.split('.')[-2]
        #print(id)
        color_raw = o3d.io.read_image(os.path.join(image_path,id+'.jpg'))
        depth_foregrounds, depth_background = convert(id, os.path.join(depth_path,id+'.png'), os.path.join(mask_path, id.zfill(5)+'.png'), output_background_path)

        # background
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_background,convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault),extrinsic=extrinsics[int(id)])
        # pcd.transform(extrinsics[int(id)])
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
        o3d.io.write_point_cloud(os.path.join(output_label_path, id+'.ply'), voxel_down_pcd)

        plydata_0 = PlyData.read(os.path.join(output_label_path, id+'.ply'))
        data_0 = plydata_0.elements[0].data
        data_pd_0 = pd.DataFrame(data_0)
        data_np_0 = np.zeros((data_pd_0.shape[0],data_pd_0.shape[1]+2), dtype=np.float64)  # 初始化储存数据的array
        property_names_0 = data_0[0].dtype.names  # 读取property的名字
        for i, name in enumerate(property_names_0):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
            data_np_0[:, i] = data_pd_0[name]


        anchor = data_np_0[:,:3]
        geo = data_np_0[:,:3]
        # color = data_np_0[:,3:6]
        feat = data_np_0[:, 3:5]
        for i in range(data_np_0.shape[0]):
            find_point = anchor[i]
            d, x = ckt.query(find_point)  # 返回最近邻点的距离d和在数组中的顺序x
            feat[i] = label[x]

        ans = np.concatenate((geo,feat),axis=1)

        # foreground
        for i in range(len(depth_foregrounds)):
            depth_f = depth_foregrounds[i][0]
            label_f = depth_foregrounds[i][1]
            label_f_instanceseg = depth_foregrounds[i][2]
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_raw, depth_f,convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault),extrinsic=extrinsics[int(id)])
            # pcd.transform(extrinsics[int(id)])
            voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
            points = np.array(voxel_down_pcd.points)
            if points.shape[0] == 0:
                continue
            o3d.io.write_point_cloud(os.path.join(output_label_path, id+'_f_'+str(i)+'.ply'), voxel_down_pcd)

            plydata_0 = PlyData.read(os.path.join(output_label_path, id+'_f_'+str(i)+'.ply'))
            data_0 = plydata_0.elements[0].data
            data_pd_0 = pd.DataFrame(data_0)
            data_np_0 = np.zeros((data_pd_0.shape[0],data_pd_0.shape[1]+2), dtype=np.float64)  # 初始化储存数据的array
            property_names_0 = data_0[0].dtype.names  # 读取property的名字
            for ii, name in enumerate(property_names_0):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
                data_np_0[:, ii] = data_pd_0[name]

            geo = data_np_0[:,:3]
            feat = data_np_0[:, 3:5]
            feat[:, 0] = label_f  # semantic label
            feat[:, 1] = max_instance_label + label_f_instanceseg  # instance label
            out = np.concatenate((geo,feat), axis=1)
            ans = np.concatenate((ans, out), axis=0)

        np.savetxt(os.path.join(output_label_path, id+'.txt'), ans)

def main(ex_path, image_path, depth_path, mask_depth, output_label_path, output_background_path, label3d_path):

    extr = o3d.io.read_pinhole_camera_trajectory(ex_path)
    extrinsics = []
    for param in extr.parameters:
        extrinsics.append(param.extrinsic)
    data_np = np.loadtxt(label3d_path)
    label = data_np[:,-2:]
    max_instance_label = np.max(label[:, -1])
    coor = data_np[:,:3]
    ckt = spt.cKDTree(coor)

    numThreads = 24
    numFrames = 300
    numFramesPerThread = np.ceil(numFrames/numThreads).astype(np.uint32)
    procs = []
    filelist = []
    for i in range(300):
        filelist.append(str(i) + ".jpg")
    for proc_index in range(numThreads):
        startIdx = proc_index*numFramesPerThread
        endIdx = min(startIdx+numFramesPerThread,numFrames)
        args = (filelist[startIdx:endIdx], image_path, depth_path, mask_depth, output_label_path, output_background_path, extrinsics, label, ckt, max_instance_label)
        proc = mlp.Process(target=scene2frame, args=args)

        proc.start()
        procs.append(proc)

    for i in range(len(procs)):
        procs[i].join()


def scene2frame_solve(ex_path, image_path, depth_path, mask_depth, output_label_path, output_background_path, label3d_path):
    try:
        main(ex_path, image_path, depth_path, mask_depth, output_label_path, output_background_path, label3d_path)
    except:
        print("[error] ", image_path)


'''
if __name__ == '__main__':
    app.run(main)
'''
