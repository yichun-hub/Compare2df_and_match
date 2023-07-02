import os
import json
import glob
import pydicom
import base36
import pandas as pd
import numpy as np
import cv2

from imutils import grab_contours
from imutils import contours

from typing import Tuple
from utils.parse_airs_annotation import get_ct_paths
from utils.parse_airs_annotation import load_json
from utils.parse_airs_annotation import get_series_uids


def slice_dict_to_volume(slice_dict: dict, shape: tuple) -> np.array:
    """ Transfer slice dict of nodule mask to a mask volume
    Arguments
    slice_dict: a dict, key is slice index and value is 2D np array mask
    shape     : the origin shape of mask volume

    Return
    a 3D mask volume
    """

    volume = np.full(shape, False)

    for slice_index, slice_image in slice_dict.items():
        volume[slice_index] = slice_image

    return volume

def decode_RLE(lst: list) -> np.array:
    """ RLE decoder
    Argument
    lst: a encoded list

    Return
    a decoded data np list
    """

    result = []

    for i, num in enumerate(lst):
        if i % 2 == 0:
            result += ([0] * num)
        else:
            result += ([1] * num)

    result = np.asarray(result)

    return result

def decode_mask(encoded_mask: str, mask_shape: tuple = (512, 512)) -> np.ndarray:
    """ Decode mask from AIRs annotation json
    Arguments
    encoded_mask: a encoded_mask string
    mask_shape  : origin shape of mask

    Return
    a np mask
    """

    mask = encoded_mask.split(',')
    mask = [base36.loads(e) for e in mask]
    mask = decode_RLE(mask)
    mask = mask.reshape(mask_shape)
    mask = mask.astype(np.bool)

    return mask



def from_dict_to_dict(origin_dict, origin_key, new_dict, new_key):
    """
    This function parse origin dict based on origin key, and create new_dict based on return value (with new_key).
    If origin dict does not have that key, return Missing_{}_Keys
    If origin dict has that key, but does not have value, return Missing_{}
    If origin dict has that key, but value as None, return Missing_{}_as_None


    """

    # ORIGIN_KEY should be part of origin_dict keys
    if origin_key in origin_dict.keys():

        if origin_dict[origin_key]!= None:
            if origin_dict[origin_key] != {}:
                new_dict[new_key] = origin_dict[origin_key]
            else:
                new_dict[new_key] = 'Missing_{}'.format(new_key)
        else:
            new_dict[new_key] = 'Missing_{}_as_None'.format(new_key)

    # if missing ORIGIN_KEY as key, fill it by 'Missing_New_KEY_as_Keys'
    else:
        new_dict[new_key] = 'Missing_{}_Keys'.format(new_key)
    return new_dict


## balanced.json 根據sliceIdx 及 compressed 比對 new_grouped_nodule_id.json --> XX

def return_nodule_info_dict_balance(json_path, batch_9 = False):
    """
    The function parse json to dict.
    
    """
    json_file = glob.glob(json_path + '/*' + '/*' + '/balanced.json')  
    if batch_9:  
        json_file = glob.glob(json_path + '/*' + '/balanced.json') 
    series_dict = {}
    for file in json_file:
        
        series_ID = str(file.split('/')[-2])
        study_ID = str(file.split('/')[-3])
        if batch_9:
            study_ID = str('None')

        json_dict = load_json(file)  
        group_dict = {}
        for i in range(len(json_dict)):
            #print(json_dict[i])
            data_dict = {}
            data_dict['studyID'] = study_ID
            data_dict['Type'] = json_dict[i]['type']
            data_dict['Lobe'] = json_dict[i]['loc']
            data_dict['Contrast'] = json_dict[i]['contrast']
            data_dict['Mask'] = json_dict[i]['mask']
            # for j in range(len(json_dict[i]['mask'])):
            #     data_dict['encoded_mask'] = json_dict[i]['mask'][j]['compressed']
            group_dict[i] = data_dict

        series_dict[series_ID] = group_dict
        #print(series_dict)

    return series_dict


def from_series_dict_to_df_balance(json_dir, ct_series_dir, series_dict, dst):
    """
    Create Dataframe from series_dict, with each annotation as rows.
    Dataframe includes following columns:

    [Patient_ID', 'Study_ID','Series_ID', 'Nodule_ID', 'Annotator', 'Annotation_ID',
    'Annotation_Action', 'Ref_Annotation_ID', 'Nodule_Type', 'Lobe', ....]

 
    """
    series_uids = get_series_uids(json_dir)    
    ct_dirs = get_ct_paths(ct_series_dir, series_uids) 

    new_data = []

    for series_id, series_value in series_dict.items(): 
        Series_ID = str(series_id)
        for idx, info in series_value.items():    
        
            Study_ID = str(info['studyID'])
            Type = info['Type']
            if Type == 'Solid':
                Type = 'S'
            elif Type == 'Part Solid':
                Type = 'PS'
            elif Type == 'Non-Solid':
                Type = 'NS'
            else:
                Type = None
            Lobe = info['Lobe']
            Contrast = info['Contrast']


            # For every slice
            if len(info['Mask']) == 0:
                SliceIdx = None
                Mask = None  
                Volume = None
  
                basic_info_lst = [Study_ID, Series_ID]
                nodule_info_lst = [Type, Lobe, Contrast, Volume]
                column_value_lst = basic_info_lst + nodule_info_lst
                new_data.append(column_value_lst) 

            else:    
                imgOrder_list = []
                mask_list = []          
                for j in range(len(info['Mask'])):
                    SliceIdx = info['Mask'][j]['sliceIdx']
                    Mask = info['Mask'][j]['compressed']

                    ## decode mask
                    mask = decode_mask(Mask)
                    mask_list.append(mask)
                    imgOrder_list.append(SliceIdx)

                    # basic_info_lst = [j, Study_ID,Series_ID]
                    # nodule_info_lst = [Type, Lobe, Contrast, SliceIdx, Mask]
                    # column_value_lst = basic_info_lst + nodule_info_lst
                    # new_data.append(column_value_lst)

                # recover mask to original shape

                ct_dir = ct_dirs[series_id]
                num_img_slice = len(glob.glob(os.path.join(ct_dir, '*.dcm')))

                mask_arr = np.array(mask_list)
                imgOrder_list = [int(i) for i in imgOrder_list]
                slice_dict = {order: mask_arr[i] for i, order in enumerate(imgOrder_list)}
                mask = slice_dict_to_volume(slice_dict, (num_img_slice, 512, 512))

                # Measure volume
                # mask = np.array(mask * 255, dtype=np.uint8)

                # volume = 0
                # for s in mask:
                #     if np.any(s):
                #         cnts = cv2.findContours(s.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #         cnts = grab_contours(cnts)
                #         cnts = contours.sort_contours(cnts)[0]
                #         for c in cnts:
                #             area = cv2.contourArea(c)
                #             volume += area

                basic_info_lst = [Study_ID, Series_ID]
                nodule_info_lst = [Type, Lobe, Contrast, mask]

                column_value_lst = basic_info_lst + nodule_info_lst

                new_data.append(column_value_lst)
    
    #return new_data

    new_df = pd.DataFrame(new_data, columns = ['Study_ID','Series_ID', 'Nodule_Type', 'Lobe', 'Contrast',  'Mask'])
    new_df.to_csv(dst)
    print(new_df)

    return new_df
                    
  

if __name__=='__main__':
    json_path = "/mnt/DATA/DATA/AIRs/Datasets/batch_11/for_brenda/json/"
    series_dict = return_nodule_info_dict_balance(json_path)  
    dst = os.path.join('./', 'AI_predict_batch_11_mask.csv')
    json_dir = "/mnt/DATA/CODE/airs_develop/airs_annotation_analysis/annotations/batch_11/V20220331150747_Bt11Ex1"
    ct_series_dir = "/mnt/DATA/DATA/AIRs/Datasets/batch_11/for_brenda/ct"
    from_series_dict_to_df_balance(json_dir, ct_series_dir,series_dict, dst) 



            
