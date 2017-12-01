import waterz
import numpy as np
import h5py
import pdb

voxel_size = np.array([4,4,40])

aff_sample = "../predictions/run_22/100000/p_0/sample_C_padded_20160501.aligned.filled.cropped.62:153.hdf"
f_aff = h5py.File(aff_sample, "r")

aff_offset = np.array([off for off in list(f_aff["volumes/labels/pred_affinities"].attrs.items())[0][1][::-1]])
print("aff_offset, physical: ", aff_offset)
aff_offset = np.array([a/b for a,b in zip(aff_offset, voxel_size)])
print("aff_offset, voxel: ", aff_offset)

gt_data_dir = '/groups/saalfeld/home/funkej/workspace/projects/caffe/run/cremi_gunpowder/01_data'
gt_sample = 'sample_C_padded_20160501.aligned.filled.cropped.62:153.hdf'

f_gt = h5py.File(gt_data_dir + "/" + gt_sample, "r")

gt_offset = np.array(([off for off in list(f_aff["volumes/raw"].attrs.items())[0][1]][::-1]))
print("gt_offset, physical: ", gt_offset)
gt_offset = np.array([a/b for a,b in zip(gt_offset, voxel_size)])
print("gt_offset, voxel: ", gt_offset)
rel_offset= np.array(aff_offset - gt_offset,dtype=int)

print(rel_offset)

gt = np.array(f_gt["volumes/labels/neuron_ids_notransparency"].value, dtype=np.uint32)

path_to_affs = "../combined_predictions/sample_C_padded_20160501.aligned.filled.cropped.62:153.r22.100000.large.hdf"

print("Load data...")
f = h5py.File(path_to_affs, 'r')
affs = f["volumes/labels/mean_pred_affinities"]
affs = affs[0:3, :, :, :]

f = h5py.File("./agg_check_crop.h5", "w")
f.create_dataset("volumes/labels/pred_affinities", data=affs)
f.create_dataset("volumes/labels/gt", data=gt[14:, 106:, 106:])


thresholds = [0.7, 0.6, 0.5, 0.4]
print("Agglomerate...")
segmentations = waterz.agglomerate(affs, thresholds, gt=gt[14:, 106:, 106:])

n = 0
for s, stat in segmentations:
	print(stat)
	f.create_dataset("volumes/labels/segmentation_{}".format(n), data=s)
	n += 1

f.close()
