# HOI4D
Please check out the HOI4D Challenge on the latest project website www.hoi4d.top !

HOI4D is a large-scale 4D egocentric dataset for category-level human-object interaction. The dataset and benchmarks were generally described in a CVPR 2022 paper:
**HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction**

[[project page](https://hoi4d.github.io/)] [[paper](https://hoi4d.github.io/HOI4D_cvpr2022.pdf)] [[supplementary](https://hoi4d.github.io/supp_cvpr2022.pdf)] [[arxiv](https://arxiv.org/pdf/2203.01577.pdf)]

## Data Organization

HOI4D is constructed by collected human-object interaction RGB-D videos and various annotations including object CAD models, action segmentation, 2D motion segmentation, 3D static scene panoptic segmentation, 4D dymanics scene panoptic segmentation, category-level object pose, and human hand pose.

The data is organized below:
```
./ZY2021080000*/H*/C*/N*/S*/s*/T*/
|--align_rgb
   |--image.mp4
|--align_depth
   |--depth_video.avi
|--objpose
   |--*.json
|--action
   |--color.json
|--3Dseg
   |--raw_pc.pcd
   |--output.log
   |--label.pcd
|--2Dseg
   |--mask
```
- ZY2021080000* refers to the camera ID.
- H* refers to human ID.
- C* refers to object class.    
```
mapping = [
    '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
    'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
    'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
]
```
- N* refers to object instance ID.
- S* refers to the room ID.
- s* refers to the room layout ID.
- T* refers to the task ID.

Please refer to ```definitions/task/task_definitions.csv``` for the definitions of C\* and T\*.

The released data list refers to ```release.txt```. (The rest of the data is temporarily kept as a testset.)
## Data Formats

### Human-object Interaction RGB-D Videos

1. First you need to install [ffmpeg](https://ffmpeg.org/).
2. Then run ``` python utils/decode.py``` to generate RGB and depth images.

### Object CAD Models

For each rigid object, we provide a single object mesh ```{object category}/{object id}.obj```. The mesh itself is the canonical frame that defines the pose of the object.

For each articulated object, we provide the articulated part meshes as well as joint annotations. We utilize [partnet_anno_system](https://github.com/daerduoCarey/partnet_anno_system) to segment articulated parts from the whole object mesh. The part hierarchy is defined in ```{object category}/{object id}/result.json```, and the part meshes involved in the part hierarchy are provided in ```{object category}/{object id}/{objs}/{part name}.obj```. We provide joint annotations ```{object category}/{object id}/mobility_v2.json``` including the origin, direction, and rotation (for revolute joint) or translation (for prismatic joint) limit for each joint axis. In addition to the CAD model annotations, we also provide the canonical frame of each articulated part ```{object category}/{object id}/{objs}/{part name}_align.obj``` that is used to define the part pose.

### Action Segmentation

We present Segmentation to record per-frame action class. ```duration```denotes the length of video, ```event```denotes the label of the clip, ```startTime and endTime```denotes the range of the clip.


### 2D Motion Segmentation

We present 2D motion segmentation to annotate the human hands and the objects related to the interaction in each video. You can first use the ```get_color_map``` function in ```utils/color_map.py``` to convert the RGB labels to indices, and then refer to ```definitions/motion segmentation/label.csv``` to correlate each index to its corresponding semantic meaning.

### 3D Static Scene Panoptic Segmentation

- ```raw_pc.pcd and label.pcd```is the raw point cloud and the label of the reconstructed static scene. The detailed definitions refer to ```definitions/3D segmentation/labels.xlsx```.

- ```output.log``` is the camera pose of each frame.

>**Note**: You can easily use the camera pose using open3d.
```python
import open3d as o3d
outCam = o3d.io.read_pinhole_camera_trajectory(output.log).parameters
```

### 4D Dynamic Scene Panoptic Segmentation

We provide scripts to generate 4D panoptic Segmentation labels. 
```python
python prepare_4Dseg/prepare_4Dseg_dataset.py --data_dir /PATH/TO/HOI4D --output_dir /PATH/TO/HOI4D_4Dseg
```
To save storage, you can delete intermediate files by:
```
rm -rf ./HOI4D_4Dseg/*/*/*/*/*/*/*/background
rm -rf ./HOI4D_4Dseg/*/*/*/*/*/*/*/foreground_*
rm -f ./HOI4D_4Dseg/*/*/*/*/*/*/*/semantic_segmentation_label/*.ply
```
The results are output in semantic_segmentation_label/*.txt. The penultimate column is the semantic label and the last column is the instance label. 

### Category-level Object Pose

- ```anno``` refers to translation of the part.
- ```rotation``` refers to rotation of the part.
- ```dimensions``` refers to scale of the part.
Take rigid objects as an example, you can easily load the object pose using following code: 

```python
from scipy.spatial.transform import Rotation as Rt
import numpy as np

def read_rtd(file, num=0):
    with open(file, 'r') as f:
        cont = f.read()
        cont = eval(cont)
    if "dataList" in cont:
        anno = cont["dataList"][num]
    else:
        anno = cont["objects"][num]

    trans, rot, dim = anno["center"], anno["rotation"], anno["dimensions"]
    trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
    rot = np.array([rot['x'], rot['y'], rot['z']])
    dim = np.array([dim['length'], dim['width'], dim['height']], dtype=np.float32)
    rot = Rt.from_euler('XYZ', rot).as_rotvec()
    return np.array(rot, dtype=np.float32), trans, dim
```

### Human Hand Pose

We present hand pose based on MANO parameters in each video. In each `.pickle` file:

- `"poseCoeff"` :  refers to 3 global rotation + 45 mano pose parameters
- `"beta"` :  refers to 10 mano shape parameters. Shape of each human ID H* are the same.
- `"trans" ` ： refers to translation of the hand in camera frame
- `"kps2D"` : refers to 21 keypoints projection coordination of rendered hand pose on each image.

Install manopth from [here](https://github.com/hassony2/manopth.git) and put the whole `manopth` folder in the same place as your code. You may have reference problems when calling MANO. Please modify the first few lines of the corresponding files according to the actual situation.

To get 3D keypoints and camera frame hand vertices, the following code might help:

```python
from manopth.manopth.manolayer import ManoLayer
import pickle
import torch

manolayer = ManoLayer(
        mano_root='manopth/mano/models', use_pca=False, ncomps=45, flat_hand_mean=True, side='right')
f = open(${PKL_PATH}, 'rb')
hand_info = pickle.load(f, encoding='latin1')
f.close()

theta = nn.Parameter(torch.FloatTensor(hand_info['poseCoeff']).unsqueeze(0))
beta = nn.Parameter(torch.FloatTensor(hand_info['beta']).unsqueeze(0))
trans = nn.Parameter(torch.FloatTensor(hand_info['trans']).unsqueeze(0))
hand_verts, hand_joints = manolayer(theta, beta)
kps3d = hand_joints / 1000.0 + trans.unsqueeze(1) # in meters
hand_transformed_verts = hand_verts / 1000.0 + trans.unsqueeze(1)
```

## Citation

Please cite HOI4D if it helps your research: 

```x
@inproceedings{liu2022hoi4d,
  title={HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction},
  author={Liu, Yunze and Liu, Yun and Jiang, Che and Lyu, Kangbo and Wan, Weikang and Shen, Hao and Liang, Boqiang and Fu, Zhoujie and Wang, He and Yi, Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21013--21022},
  year={2022}
}
```
