from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import pandas as pd
import numpy as np
from nnunet.paths import nnUNet_raw_data
from nnunet.dataset_conversion.utils import generate_dataset_json

if __name__ == '__main__':
    # this is the data folder from the kits21 github repository, see https://github.com/neheller/kits21
    kits_data_dir = '/home/aditya/External_Drive/kits21/kits21/data'
    knight_data_dir = '/home/aditya/UMN_Research/Nikos_Lab/Capstone/KNIGHT/knight/data/knight_2_labels.csv'

    # This script uses the majority voted segmentation as ground truth
    kits_segmentation_filename = 'aggregated_MAJ_seg.nii.gz'

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    task_id = 135
    task_name = "KiTS2021"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw_data, foldername)
    
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    
    imagests = join(out_base, "imagesTs")
    labelsts = join(out_base, "labelsTs")

    phitr = join(out_base,"phiTr")
    phits = join(out_base,"phiTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelsts)
    maybe_mkdir_p(phitr)
    maybe_mkdir_p(phits)

    case_ids = subdirs(kits_data_dir, prefix='case_', join=False)
    test_idx = 0.75*len(case_ids)

    df = pd.read_csv(knight_data_dir)
    df.drop(columns='Unnamed: 0',inplace=True) # Remove the extra numbering column

    for idx,c in enumerate(case_ids):
        if isfile(join(kits_data_dir, c, kits_segmentation_filename)):
            if idx >= test_idx:
                shutil.copy(join(kits_data_dir, c, kits_segmentation_filename), join(labelsts, c + '.nii.gz'))
                shutil.copy(join(kits_data_dir, c, 'imaging.nii.gz'), join(imagests, c + '.nii.gz'))
                np.save(join(phits, c + '.npy'),df.loc[idx].to_numpy())
            else:
                shutil.copy(join(kits_data_dir, c, kits_segmentation_filename), join(labelstr, c + '.nii.gz'))
                shutil.copy(join(kits_data_dir, c, 'imaging.nii.gz'), join(imagestr, c + '.nii.gz'))
                np.save(join(phitr, c + '.npy'),df.loc[idx].to_numpy())

    generate_dataset_json(join(out_base, 'dataset.json'),
                          imagestr,
                          imagests,
                          ('CT',),
                          {
                              0: 'background',
                              1: "kidney",
                              2: "tumor",
                              3: "cyst",
                          },
                          task_name,
                          license='see https://kits21.kits-challenge.org/participate#download-block',
                          dataset_description='see https://kits21.kits-challenge.org/',
                          dataset_reference='https://www.sciencedirect.com/science/article/abs/pii/S1361841520301857, '
                                            'https://kits21.kits-challenge.org/',
                          dataset_release='0')