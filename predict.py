import sys
import h5py
from gunpowder import *
from gunpowder.tensorflow import Predict
import os
import json
import numpy as np

data_dir = '/groups/saalfeld/home/funkej/workspace/projects/caffe/run/cremi_gunpowder/01_data'
sample = 'sample_C_padded_20160501.aligned.filled.cropped.62:153'

data_dir = "."
sample = 'sample_C_padded_20160501.aligned.filled.cropped.62:153.0:268.0:268'

def predict(checkpoint_file, net_io_file, output_dir):
    
    with open(net_io_file, "r") as f:
        net_io_names = json.load(f)

    voxel_size = (40,4,4)
    input_size = Coordinate((84,268,268))*voxel_size
    output_size = Coordinate((56,56,56))*voxel_size
    context = (input_size - output_size)/2

    register_volume_type('PRED_AFFINITIES')
    register_volume_type('SIGMA')
    
    chunk_request = BatchRequest()
    
    chunk_request.add(VolumeTypes.RAW, input_size, voxel_size=voxel_size)
    chunk_request.add(VolumeTypes.PRED_AFFINITIES, output_size, voxel_size=voxel_size)
    chunk_request.add(VolumeTypes.SIGMA, output_size, voxel_size=voxel_size)
    
    f=h5py.File(os.path.join(data_dir, sample + ".hdf"), "r")
    raw_roi = Roi(offset=np.array([0,0,0]), shape=np.array(f["volumes/raw"].shape) * np.array([40,4,4]))
    f.close()
 
    source = (Hdf5Source(os.path.join(data_dir, sample + ".hdf"),
                        datasets={VolumeTypes.RAW: 'volumes/raw'},
                        volume_specs = {VolumeTypes.RAW: 
					VolumeSpec(roi=raw_roi, voxel_size=(40,4,4))}) +
              Normalize())
              #Pad({VolumeTypes.RAW: (4000, 400, 400)}))

    with build(source):
        raw_spec = source.spec[VolumeTypes.RAW]

    print "raw spec roi", raw_spec.roi

    pipeline = (source + 
		IntensityScaleShift(2, -1) +
                ZeroOutConstSections() + 
                Predict(checkpoint_file,
                        inputs = {net_io_names['raw']: VolumeTypes.RAW,},
                        outputs= {net_io_names['affs']: VolumeTypes.PRED_AFFINITIES,
                                  net_io_names['sigma']: VolumeTypes.SIGMA,},
                        volume_specs = {VolumeTypes.PRED_AFFINITIES: 
                                        VolumeSpec(roi=raw_spec.roi, 
                                                   voxel_size=raw_spec.voxel_size,
                                                   dtype=np.float32),
                                        VolumeTypes.SIGMA:
                                        VolumeSpec(roi=raw_spec.roi,
                                                   voxel_size=raw_spec.voxel_size,
                                                   dtype=np.float32)}) + 
                PrintProfilingStats() + 
                Scan(chunk_request) + 
                Snapshot({VolumeTypes.RAW: 'volumes/raw',
                          VolumeTypes.PRED_AFFINITIES: 'volumes/labels/pred_affinities',
                          VolumeTypes.SIGMA: 'volumes/labels/sigma'},
                          dataset_dtypes={VolumeTypes.PRED_AFFINITIES: np.float32,
                                          VolumeTypes.SIGMA: np.float32,},
                          every=1,
                          output_dir=output_dir,
                          output_filename=sample + '.hdf')
               )

    with build(pipeline):
        raw_spec = source.spec[VolumeTypes.RAW].copy()
        aff_spec = raw_spec.copy()
        #aff_spec.roi = raw_spec.roi.grow(-context, -context)
        
        whole_request = BatchRequest({VolumeTypes.RAW: raw_spec,
                                      VolumeTypes.PRED_AFFINITIES: aff_spec,
                                      VolumeTypes.SIGMA: aff_spec})
        print("Requesting " + str(whole_request) + " in chunks of " + str(chunk_request))
        pipeline.request_batch(whole_request)

if __name__ == "__main__":
    for run in [7,8,10]:
    	checkpoint_file = "./models/run_{}/bunet_checkpoint_96000".format(run)
    	net_io_file = "./models/run_{}/net_io_names.json".format(run)
    	output_dir = "./predictions/run_{}".format(run) + "/96000/p_{}"
    	for n in range(50):
            print("Predict {}/{}".format(n, 50))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            predict(checkpoint_file, net_io_file, output_dir.format(n))
