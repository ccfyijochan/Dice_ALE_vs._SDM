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
def dice(PSI_uncor_file, ALE_uncor_file,
         PSI_cor_file=None, ALE_cor_file=None,
         PSI_uncor_file_neg=None, ALE_uncor_file_neg=None,
         AES_uncor_file=None, AES_uncor_file_neg=None, 
         PSI_cor_file_neg=None, ALE_cor_file_neg=None,
         PSI_stat_file=None, ALE_stat_file=None, 
         PSI_cor_stat_file=None, ALE_cor_stat_file=None,
         AES_stat_file=None, ALE_stat_file_neg=None, ALE_cor_stat_file_neg=None,
         study=None
         ):

    PSI_uncor_file = mask_using_nan(PSI_uncor_file, PSI_stat_file)
    ALE_uncor_file = mask_using_nan(ALE_uncor_file, ALE_stat_file)     
    if PSI_cor_file is not None:
        PSI_cor_file = mask_using_nan(PSI_cor_file, PSI_cor_stat_file)
    if ALE_cor_file is not None:
        ALE_cor_file = mask_using_nan(ALE_cor_file, ALE_cor_stat_file)
    if ALE_cor_file_neg is not None:
        ALE_cor_file_neg = mask_using_nan(ALE_cor_file_neg, ALE_cor_stat_file_neg)
    if PSI_uncor_file_neg is not None:
        PSI_uncor_file_neg = mask_using_nan(PSI_uncor_file_neg, PSI_stat_file)
    if ALE_uncor_file_neg is not None:
        ALE_uncor_file_neg = mask_using_nan(ALE_uncor_file_neg, ALE_stat_file_neg)
    if AES_uncor_file is not None:
        AES_uncor_file = mask_using_nan(AES_uncor_file, AES_stat_file)
    if AES_uncor_file_neg is not None:
       AES_uncor_file_neg = mask_using_nan(AES_uncor_file_neg, AES_stat_file)
    if PSI_cor_file_neg is not None:
        PSI_cor_file_neg = mask_using_nan(PSI_cor_file_neg, PSI_cor_stat_file)

     # *** Obtain Dice coefficient for each combination of images
     # Comparison of replication analyses
    if AES_uncor_file is not None:
        PSI_res_AES_pos_dice = sorrenson_dice(AES_uncor_file, PSI_uncor_file)
        PSI_res_AES_neg_dice = sorrenson_dice(AES_uncor_file_neg, PSI_uncor_file_neg)
        PSI_res_ALE_pos_dice = sorrenson_dice(ALE_uncor_file, PSI_uncor_file)
    if PSI_uncor_file_neg is not None:
        PSI_res_ALE_neg_dice = sorrenson_dice(ALE_uncor_file_neg, PSI_uncor_file_neg)
    
    if AES_uncor_file is not None:
        AES_res_ALE_pos_dice = sorrenson_dice(ALE_uncor_file, AES_uncor_file)
        AES_res_ALE_neg_dice = sorrenson_dice(ALE_uncor_file_neg, AES_uncor_file_neg)
    # Comparison of correction tests
    if PSI_cor_file is not None:
        PSI_res_ALE_pos_dice_cor = sorrenson_dice(PSI_cor_file, ALE_cor_file)
    if PSI_cor_file_neg is not None:
        PSI_res_ALE_neg_dice_cor = sorrenson_dice(ALE_cor_file_neg, PSI_cor_file_neg)

        # *** Printing results
    if AES_uncor_file is not None:
        print ("PSI/AES positive activation dice coefficient = %.5f, %.0f, %.0f" % PSI_res_AES_pos_dice)
        print ("PSI/ALE positive activation dice coefficient = %.5f, %.0f, %.0f" % PSI_res_ALE_pos_dice)
    if AES_uncor_file is not None:
        print ("AES/ALE positive activation dice coefficient = %.5f, %.0f, %.0f" % AES_res_ALE_pos_dice)
        print ("cor PSI/cor ALE positive activation dice coefficient = %.5f, %.0f, %.0f" % PSI_res_ALE_pos_dice_cor)
    if ALE_cor_file_neg is not None:
        print ("PSI/AES negative activation dice coefficient = %.5f, %.0f, %.0f" % PSI_res_AES_neg_dice)
        print ("PSI/ALE negative activation dice coefficient = %.5f, %.0f, %.0f" % PSI_res_ALE_neg_dice)
        print ("AES/ALE negative activation dice coefficient = %.5f, %.0f, %.0f" % AES_res_ALE_neg_dice)
        print ("cor PSI/ cor ALE negative activation dice coefficient = %.5f, %.0f, %.0f" % PSI_res_ALE_neg_dice_cor)
    elif AES_uncor_file_neg is not None:
        print ("PSI/AES negative activation dice coefficient = %.5f, %.0f, %.0f" % PSI_res_AES_neg_dice)





