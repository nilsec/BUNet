import sys
from gunpowder import *
from gunpowder.tensorflow import *
import os
import math
import json

data_dir = '/media/nilsec/Backup/cremi/20170312_mala_v2'
samples = [
    'sample_A.augmented.0',
    'sample_B.augmented.0',
    'sample_C.augmented.0'
]
samples = ['sample_C.augmented.0']

def train_until(max_iteration):

    with open('./models/net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    register_volume_type('RAW')
    register_volume_type('ALPHA_MASK')
    register_volume_type('GT_LABELS')
    register_volume_type('GT_MASK')
    register_volume_type('GT_SCALE')
    register_volume_type('GT_AFFINITIES')
    register_volume_type('PREDICTED_AFFS')
    register_volume_type('LOSS_GRADIENT')
    register_volume_type('SIGMA_LOGITS_ALE')

    input_size = Coordinate((21,268,268))*(40,4,4)
    output_size = Coordinate((21,56,56))*(40,4,4)

    request = BatchRequest()
    request.add(VolumeTypes.RAW, input_size, voxel_size=(40,4,4))
    request.add(VolumeTypes.GT_LABELS, output_size, voxel_size=(40,4,4))
    request.add(VolumeTypes.GT_MASK, output_size, voxel_size=(40,4,4))
    request.add(VolumeTypes.GT_SCALE, output_size, voxel_size=(40,4,4))
    request.add(VolumeTypes.GT_AFFINITIES, output_size, voxel_size=(40,4,4))
    
    snapshot_request = BatchRequest({
        VolumeTypes.PREDICTED_AFFS: request[VolumeTypes.GT_AFFINITIES],
        VolumeTypes.LOSS_GRADIENT: request[VolumeTypes.GT_AFFINITIES],
    })

    data_sources = tuple(
        Hdf5Source(
            os.path.join(data_dir, sample + '.hdf'),
            datasets = {
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids_notransparency',
                VolumeTypes.GT_MASK: 'volumes/labels/mask',
            }
        ) +
        Normalize() +
        RandomLocation() +
        Reject()
        for sample in samples
    )
    
    """
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
    """

    train_pipeline = (
        data_sources +
        RandomProvider() +
        GrowBoundary(steps=4, only_xy=True) +
        SplitAndRenumberSegmentationLabels() +
        AddGtAffinities([[-1,0,0], [0,-1,0], [0,0,-1]]) +
        BalanceLabels({
                VolumeTypes.GT_AFFINITIES: VolumeTypes.GT_SCALE
            },
            {
                VolumeTypes.GT_AFFINITIES: VolumeTypes.GT_MASK
            }) +
        ZeroOutConstSections() +
        Train(
            './models/bunet',
            optimizer=net_io_names['optimizer'],
            loss=net_io_names['loss'],
            inputs={
                net_io_names['raw']: VolumeTypes.RAW,
                net_io_names['gt_affs']: VolumeTypes.GT_AFFINITIES,
                net_io_names['loss_weights']: VolumeTypes.GT_SCALE
            },
            outputs={
                net_io_names['affs']: VolumeTypes.PREDICTED_AFFS
            },
            gradients={
                net_io_names['affs']: VolumeTypes.LOSS_GRADIENT
            }) +
        PrintProfilingStats(every=10)
    )
    """ 
        Snapshot({
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                VolumeTypes.GT_AFFINITIES: 'volumes/labels/affinities',
                VolumeTypes.PREDICTED_AFFS: 'volumes/labels/pred_affinities',
                VolumeTypes.LOSS_GRADIENT: 'volumes/loss_gradient',
            },
            every=100,
            output_filename='./models/bunet_logs/run_10/batch_{iteration}.hdf',
            additional_request=snapshot_request) +
        PrintProfilingStats(every=10)
        
    )
    """

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":
    set_verbose(False)
    iteration = int(sys.argv[1])
    train_until(iteration)
