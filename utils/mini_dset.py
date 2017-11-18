import h5py

sample = "/groups/saalfeld/home/funkej/workspace/projects/caffe/run/cremi_gunpowder/01_data/sample_C_padded_20160501.aligned.filled.cropped.62:153.hdf"


f = h5py.File(sample, "r")
raw = f["volumes/raw"][:,0:268,0:268]
f.close()

f = h5py.File("./sample_C_padded_20160501.aligned.filled.cropped.62:153.0:268.0:268.hdf", "w")
f.create_dataset("volumes/raw", data=raw)
f.close()


f = h5py.File("./sample_C_padded_20160501.aligned.filled.cropped.62:153.0:268.0:268.hdf", "r")
print f["volumes/raw"].shape
f.close()

