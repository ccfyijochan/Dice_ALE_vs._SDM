import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import scipy
import os
import warnings

def sorrenson_dice(data1_file, data2_file, reslice=True):
    # Load nifti images
    data1_img = nib.load(data1_file)
    data2_img = nib.load(data2_file)

    # Load data from images
    data2 = data2_img.get_data()
    data1 = data1_img.get_data()

    # Get asbolute values (positive and negative blobs are of interest)
    data2 = np.absolute(data2)
    data1 = np.absolute(data1)

    if reslice:
        # Resample data1 on data2 using nearest nneighbours
        data1_resl_img = resample_from_to(data1_img, data2_img, order=0)
        # Load data from images
        data1_res = data1_resl_img.get_data()
        data1_res = np.absolute(data1_res)
            
        # Resample data2 on data1 using nearest nneighbours
        data2_resl_img = resample_from_to(data2_img, data1_img, order=0)        
        data2_res = data2_resl_img.get_data()
        data2_res = np.absolute(data2_res)

    # Masking (compute Dice using intersection of both masks)
    if reslice:
        background_1 = np.logical_or(np.isnan(data1), np.isnan(data2_res))
        background_2 = np.logical_or(np.isnan(data1_res), np.isnan(data2))

        data1 = np.nan_to_num(data1)
        data1_res = np.nan_to_num(data1_res)
        data2 = np.nan_to_num(data2)
        data2_res = np.nan_to_num(data2_res)

        num_activated_1 = np.sum(data1 > 0)
        num_activated_res_1 = np.sum(data1_res>0)
        num_activated_2 = np.sum(data2>0)
        num_activated_res_2 = np.sum(data2_res>0)

        dark_dice_1 = np.zeros(2)
        if num_activated_1 != 0:
            dark_dice_1[0] = np.sum(data1[background_1]>0).astype(float)/num_activated_1*100
        if num_activated_res_1 != 0:
            dark_dice_1[1] = np.sum(data1_res[background_2]>0).astype(float)/num_activated_res_1*100

        dark_dice_2 = np.zeros(2)
        if num_activated_2 != 0:
            dark_dice_2[0] = np.sum(data2[background_2]>0).astype(float)/num_activated_2*100
        if num_activated_res_2 != 0:
            dark_dice_2[1] = np.sum(data2_res[background_1]>0).astype(float)/num_activated_res_2*100

        data1[background_1] = 0
        data2_res[background_1] = 0

        data1_res[background_2] = 0
        data2[background_2] = 0
    else:
        background = np.logical_or(np.isnan(data1), np.isnan(data2))

        data1 = np.nan_to_num(data1)
        data2 = np.nan_to_num(data2)

        num_activated_1 = np.sum(data1 > 0)
        num_activated_2 = np.sum(data2>0)

        dark_dice = np.zeros(2)
        if num_activated_1 !=0:
            dark_dice[0] = np.sum(data1[background]>0).astype(float)/num_activated_1*100

        if num_activated_2 !=0:
            dark_dice[1] = np.sum(data2[background]>0).astype(float)/num_activated_2*100

        data1[background] = 0
        data2[background] = 0

    # Vectorize
    data1 = np.reshape(data1, -1)
    data2 = np.reshape(data2, -1)
    if reslice:
        data1_res = np.reshape(data1_res, -1)
        data2_res = np.reshape(data2_res, -1)

    if reslice:
        dice_res_1 = 1-scipy.spatial.distance.dice(data1_res>0, data2>0)
        dice_res_2 = 1-scipy.spatial.distance.dice(data1>0, data2_res>0)

        if not np.isclose(dice_res_1, dice_res_2, atol=0.01):
            warnings.warn("Resliced 1/2 and 2/1 dices are not close")

        if not np.isclose(dark_dice_1[0], dark_dice_1[1], atol=0.01):
            warnings.warn("Resliced 1/2 and 2/1 dark dices 1 are not close")

        if not np.isclose(dark_dice_2[0], dark_dice_2[1], atol=0.01):
            warnings.warn("Resliced 1/2 and 2/1 dark dices 2 are not close")

        dices = (dice_res_1, dark_dice_1[1], dark_dice_2[1])
    else:
        dices = (1-scipy.spatial.distance.dice(data1>0, data2>0), dark_dice[0], dark_dice[1])
    
    return dices


def mask_using_nan(data_file, mask_file, filename=None):
    # Set masking using NaN's
    data_img = nib.load(data_file)
    data_orig = data_img.get_data()

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_data()

    if np.any(np.isnan(mask_data)):
        # mask already using NaN
        mask_data_nan = mask_data
    else:
        # Replace zeros by NaNs
        mask_data_nan = mask_data.astype(float)
        mask_data_nan[mask_data_nan == 0] = np.nan

    # If there are NaNs in data_file remove them (to mask using mask_file only)
    data_orig = np.nan_to_num(data_orig)

    # Replace background by NaNs
    data_nan = data_orig.astype(float)
    data_nan[np.isnan(mask_data_nan)] = np.nan

    # Save as image
    data_img_nan = nib.Nifti1Image(data_nan, data_img.get_affine())
    if filename is None:
        filename = data_file.replace('.nii', '_nan.nii')

    nib.save(data_img_nan, filename)

    return(filename)
def dice(including_ALE_file, excluding_ALE_file, including_AES_file, excluding_AES_file, including_PSI_file, excluding_PSI_file,
         including_ALE_file_neg=None, excluding_ALE_file_neg=None, 
         including_ALE_stat_file=None, excluding_ALE_stat_file=None, excluding_ALE_stat_file_neg=None, including_ALE_stat_file_neg=None,
         including_AES_file_neg=None, excluding_AES_file_neg=None, 
         including_AES_stat_file=None, excluding_AES_stat_file=None, excluding_AES_stat_file_neg=None, including_AES_stat_file_neg=None,
         including_PSI_file_neg=None, excluding_PSI_file_neg=None, 
         including_PSI_stat_file=None, excluding_PSI_stat_file=None, excluding_PSI_stat_file_neg=None, including_PSI_stat_file_neg=None,
         study=None
         ):

    including_ALE_file = mask_using_nan(including_ALE_file, including_ALE_stat_file)  
    excluding_ALE_file = mask_using_nan(excluding_ALE_file, excluding_ALE_stat_file)    
    if including_ALE_file_neg is not None:
        including_ALE_file_neg = mask_using_nan(including_ALE_file_neg, including_ALE_stat_file_neg)
    if excluding_ALE_file_neg is not None:
        excluding_ALE_file_neg = mask_using_nan(excluding_ALE_file_neg, excluding_ALE_stat_file_neg)
    

     # *** Obtain Dice coefficient for each combination of images
     # Comparison of replication analyses
    
    if excluding_ALE_file is not None:
        excluding_ALE_res_including_ALE_pos_dice = sorrenson_dice(including_ALE_file, excluding_ALE_file)
        excluding_ALE_res_including_ALE_neg_dice = sorrenson_dice(including_ALE_file_neg, excluding_ALE_file_neg)



    if including_AES_file_neg is not None:
        including_AES_file_neg = mask_using_nan(including_AES_file_neg, including_AES_stat_file_neg)
    if excluding_AES_file_neg is not None:
        excluding_AES_file_neg = mask_using_nan(excluding_AES_file_neg, excluding_AES_stat_file_neg)
    

     # *** Obtain Dice coefficient for each combination of images
     # Comparison of replication analyses
    
    if excluding_AES_file is not None:
        excluding_AES_res_including_AES_pos_dice = sorrenson_dice(including_AES_file, excluding_AES_file)
        excluding_AES_res_including_AES_neg_dice = sorrenson_dice(including_AES_file_neg, excluding_AES_file_neg)


    if including_PSI_file_neg is not None:
        including_PSI_file_neg = mask_using_nan(including_PSI_file_neg, including_PSI_stat_file_neg)
    if excluding_PSI_file_neg is not None:
        excluding_PSI_file_neg = mask_using_nan(excluding_PSI_file_neg, excluding_PSI_stat_file_neg)
    

     # *** Obtain Dice coefficient for each combination of images
     # Comparison of replication analyses
    
    if excluding_PSI_file is not None:
        excluding_PSI_res_including_PSI_pos_dice = sorrenson_dice(including_PSI_file, excluding_PSI_file)
        excluding_PSI_res_including_PSI_neg_dice = sorrenson_dice(including_PSI_file_neg, excluding_PSI_file_neg)
   
    

        # *** Printing results
    if excluding_ALE_file is not None:
        print ("excluding_ALE/including_ALE positive activation dice coefficient = %.5f, %.0f, %.0f" % excluding_ALE_res_including_ALE_pos_dice)
    if including_ALE_file_neg is not None:
        print ("excluding_ALE/including_ALE negative activation dice coefficient = %.5f, %.0f, %.0f" % excluding_ALE_res_including_ALE_neg_dice)
    if excluding_AES_file is not None:
        print ("excluding_AES/including_AES positive activation dice coefficient = %.5f, %.0f, %.0f" % excluding_AES_res_including_AES_pos_dice)
    if including_AES_file_neg is not None:
        print ("excluding_AES/including_AES negative activation dice coefficient = %.5f, %.0f, %.0f" % excluding_AES_res_including_AES_neg_dice)
    if excluding_PSI_file is not None:
        print ("excluding_PSI/including_PSI positive activation dice coefficient = %.5f, %.0f, %.0f" % excluding_PSI_res_including_PSI_pos_dice)
    if including_PSI_file_neg is not None:
        print ("excluding_PSI/including_PSI negative activation dice coefficient = %.5f, %.0f, %.0f" % excluding_PSI_res_including_PSI_neg_dice)
    elif excluding_PSI_file_neg is not None:
        print ("excluding_ALE/including_ALE negative activation dice coefficient = %.5f, %.0f, %.0f" % excluding_ALE_res_including_ALE_neg_dice)





