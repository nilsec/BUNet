import neuroglancer
import numpy as np
import h5py
from scipy.special import expit

def view(run, 
         batch, 
         raw=True, 
         pred_aff=True, 
         gt_affs=False, 
         gt_labels=False,
         sigma=False,
         sigma_scaled=False,
         pred=False,
         voxel_size=[40,4,4]):

    path = "/media/nilsec/Backup/snapshots/run_{}/batch_{}.0.hdf".format(run, batch)
    f = h5py.File(path)
    viewer = neuroglancer.Viewer()

    
    print f["volumes/labels/pred_affinities"].attrs.modify("resolution", (1,1,1))
    print f["volumes/labels/pred_affinities"].attrs.modify("offset", np.array([0, 424/4, 424/4]))
    print f["volumes/labels/sigma"].attrs.modify("offset", np.array([0, 424/4, 424/4]))
    
    if sigma_scaled:
        sigma_scaled_arr = expit(np.reshape(f["volumes/labels/sigma"].value, np.shape(f["volumes/labels/pred_affinities"])))
        try:
            f.create_dataset("volumes/labels/sigma_scaled", data=sigma_scaled_arr)
        except RuntimeError:
            pass
        
        f["volumes/labels/sigma_scaled"].attrs.modify("offset", np.array([0,424/4,424/4]))

    shader = """
            void main() {
            float val = getDataValue();
            vec4 color= vec4(0.,0.,0., 0);
            if (val>0.55) color = vec4(val, 0,0, 1);
            emitRGBA(color);
            }
            """ 
    if raw:
        viewer.add(np.array(f["volumes/raw"].value, dtype=np.float32), name="raw")
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
    view(1, 744101, pred_aff=True, sigma_scaled=True)

