# HOI4D

HOI4D is a large-scale 4D egocentric dataset for category-level human-object interaction. The dataset and benchmarks were generally described in a CVPR 2022 paper:

**HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction**

[[project page](https://hoi4d.github.io/)] [[paper](https://hoi4d.github.io/HOI4D_cvpr2022.pdf)] [[supplementary](https://hoi4d.github.io/supp_cvpr2022.pdf)] [[arxiv](https://arxiv.org/pdf/2203.01577.pdf)]

## Data Organization

HOI4D is constructed by collected human-object interaction RGB-D videos and various annotations including object CAD models, action segmentation, 2D motion segmentation, 3D static scene panoptic segmentation, 4D dymanics scene panoptic segmentation, category-level object pose, and human hand pose.

The data is organized below:

TBD

```x
```

## Data Formats

### Human-object Interaction RGB-D Videos

TBD

### Object CAD Models

For each rigid object, we provide a single object mesh ```{object category}/{object id}.obj```. The mesh itself is the canonical frame that defines the pose of the object.

For each articulated object, we provide the articulated part meshes as well as joint annotations. We utilize [partnet_anno_system](https://github.com/daerduoCarey/partnet_anno_system) to segment articulated parts from the whole object mesh. The part hierarchy is provided in ```{object category}/{object id}/result.json```, and the part meshes involved in the part hierarchy are provided in the folder ```{object category}/{object id}/{objs}/{part name}.obj```. We provide joint annotations ```{object category}/{object id}/mobility_v2.json``` including the origin, direction, and rotation (for revolute joint) or translation (for prismatic joint) limit for each joint axis. In addition to the CAD model annotations, we also provide the canonical frame of each articulated part ```{object category}/{object id}/{objs}/{part name}_align.obj``` that is used to define the part pose.

### Action Segmentation

TBD

### 2D Motion Segmentation

We present 2D motion segmentation to annotate the human hands and the objects related to the interaction in each video. You can first use the ```get_color_map``` function in ```utils/get_color_map.py``` to convert the RGB labels to indices, and then refer to ```definitions/2D Motion Segmentation.csv``` to correlate each index to its corresponding semantic meaning.

### 3D Static Scene Panoptic Segmentation

TBD

### 4D Dynamic Scene Panoptic Segmentation

TBD

### Category-level Object Pose

TBD

### Human Hand Pose

TBD

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

## Lisence

TBD

