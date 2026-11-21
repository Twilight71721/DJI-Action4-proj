"""View bins describe observer direction in object coordinates, not image left/right."""
import numpy as np
from torch.utils.data import Subset

FACE_NAMES = ('X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-')


def view_bin(rotation,translation,center):
    rotation=np.asarray(rotation,float).reshape(3,3)
    vector=-rotation.T@np.asarray(translation,float).reshape(3)-np.asarray(center,float)
    if not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-9:
        raise ValueError('Invalid observer direction')
    axis=int(np.argmax(abs(vector)))
    return axis*2+int(vector[axis]<0)


def dataset_view_bins(dataset):
    labels=[]
    for i in range(len(dataset)):
        base,index=dataset,i
        while hasattr(base,'dataset'):
            if isinstance(base,Subset):index=base.indices[index]
            base=base.dataset
        scene,frame,instance=base.instances[index]
        gt,_,_=base._load_scene(scene)
        record=gt[str(frame)][instance]
        labels.append(view_bin(record['cam_R_m2c'],np.asarray(record['cam_t_m2c'])*base.pose_scale,
            (base.model.min_xyz_mm+base.model.max_xyz_mm)/2))
    return np.asarray(labels,int)


def view_weights(labels,cap=3.):
    if not np.isfinite(cap) or cap<1:raise ValueError('View weight cap must be >= 1')
    labels=np.asarray(labels,int)
    counts=np.bincount(labels,minlength=6)
    weights=np.minimum(np.sqrt(counts.max()/np.maximum(counts,1)),cap)
    return weights[labels],counts
