import neuroglancer
import numpy as np
import h5py
from scipy.special import expit

def view(path,
         raw=True, 
         pred_aff=True, 
         gt_affs=False, 
         gt_labels=False,
         sigma=False,
         sigma_scaled=False,
         pred=False,
         voxel_size=[40,4,4]):

    f = h5py.File(path)
    viewer = neuroglancer.Viewer()

    f["volumes/labels/pred_affinities"].attrs.modify("resolution", (40,4,4))
    f["volumes/raw"].attrs.modify("resolution", (40,4,4))
    print f["volumes/raw"].attrs.items()
    print f["volumes/labels/pred_affinities"].attrs.items()
    if sigma_scaled:
        sigma_scaled_arr = expit(np.reshape(f["volumes/labels/sigma"].value, np.shape(f["volumes/labels/pred_affinities"])))
        try:
            f.create_dataset("volumes/labels/sigma_scaled", data=sigma_scaled_arr)
        except RuntimeError:
            pass
        
        f["volumes/labels/sigma_scaled"].attrs.modify("resolution", (40,4,4))
        f["volumes/labels/sigma_scaled"].attrs.modify("offset", (0,8,8))
  

    if raw:
        raw32 = np.array(f["volumes/raw"].value, dtype=np.float32)
        try:
            f.create_dataset("volumes/raw32", data=raw32)
        except RuntimeError:
            pass
        f["volumes/raw32"].attrs.modify("resolution", (40,4,4))

    shader = """
            void main() {
            float val = getDataValue();
            vec4 color= vec4(0.,0.,0., 0);
            if (val>0.55) color = vec4(val, 0,0, 1);
            emitRGBA(color);
            }
            """ 
    if raw:
        viewer.add(f["volumes/raw32"], name="raw")
    if pred_aff:
        viewer.add(f["volumes/labels/pred_affinities"])
    if gt_affs:
        viewer.add(f["volumes/labels/affinities"])
    if gt_labels:
        viewer.add(f["volumes/labels/neuron_ids"])
    if sigma_scaled:
        viewer.add(f["volumes/labels/sigma_scaled"], name="sigma", shader=shader)
    if sigma:
        viewer.add(f["volumes/labels/sigma"], name="sigma", shader=shader)

    print viewer
   

if __name__ == "__main__":
    view("/media/nilsec/Backup/predictions_mc_2/sample_A/sample_A.augmented.0.hdf", pred_aff=True, sigma_scaled=True)

