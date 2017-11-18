from __future__ import print_function

import sys
import numpy as np
from gunpowder import *
from gunpowder.tensorflow import *
import os
import math
import json

data_dir = '/groups/saalfeld/home/funkej/workspace/projects/caffe/run/cremi_gunpowder/01_data'
samples = ['sample_A_padded_20160501.aligned.filled.cropped',
	   'sample_B_padded_20160501.aligned.filled.cropped',
	   'sample_C_padded_20160501.aligned.filled.cropped.0:90'
	  ]

affinity_neighborhood = np.array([

    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],

    [-2, 0, 0],
    [0, -3, 0],
    [0, 0, -3],

    [-3, 0, 0],
    [0, -9, 0],
    [0, 0, -9],

    [-4, 0, 0],
    [0, -27, 0],
    [0, 0, -27]
])

def train_until(max_iteration, run):
    print("Train run {} until {}...".format(run, max_iteration))

    with open('./models/run_{}/net_io_names.json'.format(run), 'r') as f:
        net_io_names = json.load(f)

    register_volume_type('RAW')
    register_volume_type('ALPHA_MASK')
    register_volume_type('GT_LABELS')
    register_volume_type('GT_MASK')
    register_volume_type('GT_SCALE')
    register_volume_type('GT_AFFINITIES')
    register_volume_type('GT_AFFINITIES_MASK')
    register_volume_type('PREDICTED_AFFS')
    register_volume_type('LOSS_GRADIENT')

    input_size = Coordinate((84,268,268))*(40,4,4)
    output_size = Coordinate((56,56,56))*(40,4,4)

    request = BatchRequest()
    request.add(VolumeTypes.RAW, input_size, voxel_size=(40,4,4))
    request.add(VolumeTypes.GT_LABELS, output_size, voxel_size=(40,4,4))
    request.add(VolumeTypes.GT_MASK, output_size, voxel_size=(40,4,4))
    request.add(VolumeTypes.GT_SCALE, output_size, voxel_size=(40,4,4))
    request.add(VolumeTypes.GT_AFFINITIES, output_size, voxel_size=(40,4,4))
    request.add(VolumeTypes.GT_AFFINITIES_MASK, output_size, voxel_size=(40,4,4))

    snapshot_request = BatchRequest()
    snapshot_request.add(VolumeTypes.RAW, input_size, voxel_size=(40,4,4))
    snapshot_request.add(VolumeTypes.PREDICTED_AFFS, output_size, voxel_size=(40,4,4))
#    snapshot_request.add(VolumeTypes.LOSS_GRADIENT, output_size, voxel_size=(40,4,4))    
    snapshot_request.add(VolumeTypes.GT_LABELS, output_size, voxel_size=(40,4,4))
    snapshot_request.add(VolumeTypes.GT_AFFINITIES, output_size, voxel_size=(40,4,4))

    data_sources = tuple(
        Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = {
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids_notransparency',
                VolumeTypes.GT_MASK: 'volumes/labels/mask',
            },
            volume_specs = {VolumeTypes.RAW: VolumeSpec(voxel_size=(40,4,4)),
                            VolumeTypes.GT_LABELS: VolumeSpec(voxel_size=(40,4,4)),
                            VolumeTypes.GT_MASK: VolumeSpec(interpolatable=False, voxel_size=(40,4,4))}
        ) +
        Normalize() +
        RandomLocation() +
        Reject()
        for sample in samples
    )
    
    
    artifact_source = (
        Hdf5Source(
            os.path.join(data_dir, 'sample_ABC_padded_20160501.defects.hdf'),
            datasets = {
                VolumeTypes.RAW: 'defect_sections/raw',
                VolumeTypes.ALPHA_MASK: 'defect_sections/mask',
            },
            volume_specs = {
                VolumeTypes.RAW: VolumeSpec(voxel_size=(40, 4, 4)),
                VolumeTypes.ALPHA_MASK: VolumeSpec(voxel_size=(40, 4, 4)),
            }
        ) +
        RandomLocation(min_masked=0.05, mask_volume_type=VolumeTypes.ALPHA_MASK) +
        Normalize() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0], subsample=8) +
        SimpleAugment(transpose_only_xy=True)
    )


    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0], prob_slip=0.05,prob_shift=0.05,max_misalign=10, subsample=8) +
        SimpleAugment(transpose_only_xy=True) +
        GrowBoundary(steps=1, only_xy=True) +
	SplitAndRenumberSegmentationLabels() + 
        AddGtAffinities(affinity_neighborhood, 
			gt_labels_mask=VolumeTypes.GT_MASK) +
        BalanceLabels(
	    labels=VolumeTypes.GT_AFFINITIES,
            scales=VolumeTypes.GT_SCALE,
            mask=VolumeTypes.GT_AFFINITIES_MASK,
            slab=(3, -1, -1, -1)
             ) +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(
            prob_missing=0.03,
            prob_low_contrast=0.01,
            prob_artifact=0.03,
            artifact_source=artifact_source,
            contrast_scale=0.5) +
        IntensityScaleShift(2,-1) +
        ZeroOutConstSections() +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(
            './models/run_{}/bunet'.format(run),
            optimizer=net_io_names['optimizer'],
            loss=net_io_names['loss'],
            inputs={
                net_io_names['raw']: VolumeTypes.RAW,
                net_io_names['gt_affs']: VolumeTypes.GT_AFFINITIES,
                net_io_names['loss_weights']: VolumeTypes.GT_SCALE
            },
            outputs={
                net_io_names['affs']: VolumeTypes.PREDICTED_AFFS,
            },
            gradients={
                net_io_names['affs']: VolumeTypes.LOSS_GRADIENT
            }) +
	IntensityScaleShift(0.5, 0.5) + 
        Snapshot({
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                VolumeTypes.GT_AFFINITIES: 'volumes/labels/affinities',
                VolumeTypes.PREDICTED_AFFS: 'volumes/labels/pred_affinities',
            },
            every=5000,
            output_filename='run_{}/'.format(run) + 'batch_{iteration}.hdf',
            additional_request=snapshot_request) + 
	PrintProfilingStats(every=10)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":
    set_verbose(False)
    iteration = int(sys.argv[1])
    run = int(sys.argv[2])
    train_until(iteration, run)
