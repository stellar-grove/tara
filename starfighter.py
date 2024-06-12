# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 07:02:54 2023

@author: DanielKorpon
"""

# --------------------------------------------------------------

sg_repo = "C:/stellar-grove"
import sys;sys.path.append(sg_repo)
import tara.stuffs as stuffs
import numpy as np
import pydicom

class MedicalImaging(object):
    
    def __init__(self,imaging_type) -> None:
        self.imaging_type = imaging_type 
        self.data = {}
        self.config = {"data_folder":f'{stuffs.udemyWD}Deep Learning with PyTorch for Medical Image Analysis/AI-IN-MEDICAL-MATERIALS/'}


    def read_files(self,path_to_folder):
        all_files = list(path_to_folder.glob("*"))
        folder_data = []
        for path in all_files:
            data = pydicom.read_file(path)
            folder_data.append(data)
        folder_data_sorted = sorted(folder_data, key=lambda slice: slice.SliceLocation)

    spyder_text = """

                dk_repo = "C:/repo/bitstaemr";sg_repo = "C:/stellar-grove"
                import sys;sys.path.append(sg_repo)
                #sys.path.append(dk_repo)

                import tara.starfighter as sf
                import pydicom
                from pathlib import Path
                import matplotlib.pyplot as plt
                import nibabel as nib
                import SimpleITK as sitk

                mi.config

                file = "ID_0000_AGE_0060_CONTRAST_1_CT.dcm"
                dicom_file = pydicom.read_file(file)



                mi = sf.MedicalImaging('')

                data_folder = mi.config['data_folder']




                path_to_head_mri = Path(f'{data_folder}03-Data-Formats/SE000001')
                all_files = list(path_to_head_mri.glob("*"))

                mri_data = []
                for path in all_files:
                    data = pydicom.read_file(path)
                    mri_data.append(data)

                for slice in mri_data[:5]:
                    print(slice.SliceLocation)

                mri_data_ordered = sorted(mri_data, key=lambda slice: slice.SliceLocation)

                full_volume = []
                for slice in mri_data_ordered:
                    full_volume.append(slice.pixel_array)

                fig, axis = plt.subplots(3, 3, figsize=(10,10))
                slice_counter = 0
                for i in range(3):
                    for j in range(3):
                        axis[i][j].imshow(full_volume[slice_counter], cmap='gray')
                        slice_counter+=1
                        
                series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path_to_head_mri))
                series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path_to_head_mri))

                series_reader = sitk.ImageSeriesReader()
                series_reader.SetFileNames(series_file_names)
                image_data = series_reader.Execute()
                head_mri = sitk.GetArrayFromImage(image=image_data)


                fig, axis = plt.subplots(3, 3, figsize=(10,10))
                slice_counter = 0
                for i in range(3):
                    for j in range(3):
                        axis[i][j].imshow(head_mri[slice_counter], cmap='gray')
                        slice_counter+=1

                """