import os
import numpy as np
import argparse
import multiprocessing as mlp
from scene2frame import scene2frame_solve


def solve1(data_dir, data_2Dseg_dir, output_dir):
    ex_path = os.path.join(data_dir, "3Dseg", "output.log")
    image_path = os.path.join(data_dir, "align_rgb")
    depth_path = os.path.join(data_dir, "align_depth")
    mask_path = os.path.join(data_2Dseg_dir, "mask")
    output_label_path = os.path.join(output_dir, "semantic_segmentation_label")
    output_background_path = os.path.join(output_dir, "background")
    label3d_path = os.path.join(data_dir, "3Dseg", "label.txt")
    scene2frame_solve(ex_path, image_path, depth_path, mask_path, output_label_path, output_background_path, label3d_path)


def prepare(startIdx, endIdx, output_dir, npz_path):
    for i in range(startIdx, endIdx):
        pc = []
        rgb = []
        semantic = []
        with open(os.path.join(output_dir, "semantic_segmentation_label", str(i) + ".txt"), 'r') as f:
            for line in f:
                line = line.split(' ')
                pc.append([float(line[0]), float(line[1]), float(line[2])])
                rgb.append([float(line[0]), float(line[1]), float(line[2])])
                semantic.append(int(float(line[3]) + 0.01))
        pc = np.array(pc)
        rgb = np.array(rgb)
        semantic = np.array(semantic)
        center = np.mean(pc, axis=0)
        np.savez(os.path.join(npz_path, str(i) + ".npz"), pc=pc, rgb=rgb, semantic=semantic, center=center)


def solve2(output_dir):
    npz_path = os.path.join(output_dir, "npz")
    os.makedirs(npz_path, exist_ok=True)
    numThreads = 24
    numFrames = 300
    numFramesPerThread = np.ceil(numFrames/numThreads).astype(np.uint32)
    procs = []
    for proc_index in range(numThreads):
        startIdx = proc_index*numFramesPerThread
        endIdx = min(startIdx+numFramesPerThread,numFrames)
        args = (startIdx, endIdx, output_dir, npz_path)
        proc = mlp.Process(target=prepare, args=args)

        proc.start()
        procs.append(proc)

    for i in range(len(procs)):
        procs[i].join()


def prepare_4Dseg_data(data_dir, output_dir, datalist_path):

    datalist = []
    with open(datalist_path, "r") as f:
        for line in f:
            content = line.strip()
            if len(content) == 0:
                continue
            datalist.append(content)
    
    cnt = 0
    for video_name in datalist:
        has_file = 0
        for i in range(300):
            if os.path.isfile(os.path.join(output_dir, video_name, "npz", str(i) + ".npz")):
                has_file += 1
        if has_file == 300:
            cnt += 1
        # else:
        #     print(video_name)
    print(cnt)
    
    
    # Step1:
    print("------start step 1: get semantic_segmentation_label------")
    for video_name in datalist:
        print(video_name)

        cnt = 0
        data_2Dseg_dir = None
        for seg2D_dir in data_dir:
            if os.path.isdir(os.path.join(seg2D_dir, video_name, "refine_2Dseg")):
                data_2Dseg_dir = seg2D_dir
                cnt += 1
        assert(cnt == 1)
        solve1(os.path.join(data_dir, video_name), os.path.join(data_2Dseg_dir, video_name), os.path.join(output_dir, video_name))
    print("------finish step 1!------")

    '''
    # Step2: 
    print("------start step 2: get npz------")
    for video_name in datalist:
        print(video_name)
        solve2(os.path.join(output_dir, video_name))
    print("------finish step 2!------")
    '''


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="/PATH/TO/HOI4D")
    parser.add_argument('--output_dir', type=str, default="/PATH/TO/HOI4D_4Dseg")
    parser.add_argument('--train_datalist_path', type=str, default="./datalists/train_all.txt")
    parser.add_argument('--test_datalist_path', type=str, default="./datalists/test_all.txt")

    args = parser.parse_args()

    print("start preparing training set")
    prepare_4Dseg_data(args.data_dir, args.output_dir, args.train_datalist_path)
    print("start preparing test set")
    prepare_4Dseg_data(args.data_dir, args.output_dir, args.test_datalist_path)


if __name__ == "__main__":
    main()
