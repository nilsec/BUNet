import matplotlib.pyplot as plt
import h5py
import numpy as np

def view_snap(data_path, snap_id, epoch):
    f = h5py.File(data_path, "r")
    snaps = f["snapshot_%s" % snap_id]

    affs = snaps["affs"]
    affs_noise = snaps["affs_noise"]
    sigma = snaps["sigma"]
    gt_affs = snaps["gt_affs"]

    f, axes = plt.subplots(4,2)
    axes[0,0].imshow(affs[epoch, 0,0,:,:])
    axes[0,1].imshow(affs[epoch, 1,0,:,:])

    axes[1,0].imshow(affs_noise[epoch, 0,0,:,:])
    axes[1,1].imshow(affs_noise[epoch, 1,0,:,:])

    axes[2,0].imshow(np.abs(sigma[epoch, 0,0,:,:]))
    axes[2,1].imshow(np.abs(sigma[epoch, 1,0,:,:]))

    axes[3,0].imshow(gt_affs[epoch, 0,0,:,:])
    axes[3,1].imshow(gt_affs[epoch, 1,0,:,:])
 
    plt.show()

if __name__ == "__main__":
    view_snap("/media/nilsec/d0/bunet/square_data_noisy/snaps.hdf5",
              8,
              -1) 
