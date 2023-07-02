
import os
import json
import glob
import pydicom

import pandas as pd
import numpy as np
import base36
import cv2
import itertools
import numba as nb

from imutils import grab_contours
from imutils import contours
from typing import Tuple
from utils.parse_airs_annotation import get_ct_paths
from utils.parse_airs_annotation import load_json
from utils.parse_airs_annotation import get_series_uids


from new_balanced_json_to_df import return_nodule_info_dict_balance, from_series_dict_to_df_balance

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

# For new_grouped_nodule_id.json
def return_nodule_info_dict(json_dict):
    """
    The function parse json to dict.
    The json file contains the Group ID which can match the other json file without Group ID.

    """

    series_dict = {}
    for series_id, views in json_dict.items():
        
        group_dict = {}
        for group, content in views.items():
            #print(group)
            for anno_content, data in content["annotation_contents"].items():
            

                data_dict = {}
                data_dict['patient_id'] = data['Annotation_PatientId']
                data_dict['study_id'] = data['Annotation_StudyInstanceUid']
                #data_dict['series_id'] = data['Annotation_SeriesInstanceUid']
                data_dict['annotation_id'] = data['Annotation_Id']
                data_dict['annotation_label'] = data['Annotation_Label']
                data_dict['doctor'] = data['Annotation_UserName']
                data_dict['annotation_action'] = data['Annotation_Action']
                data_dict['ref_annotation_id'] = data['Annotation_RefAnnotationId']

                # check for annotation_content
                if data['Annotation_Content'] != 'None':
                    data_dict = from_dict_to_dict(data['Annotation_Content'], 'type', data_dict, 'nodule_type')
                    data_dict = from_dict_to_dict(data['Annotation_Content'], 'lobe', data_dict, 'lobe')
                    data_dict = from_dict_to_dict(data['Annotation_Content'], 'path', data_dict, 'pathology')
                    data_dict = from_dict_to_dict(data['Annotation_Content'], 'malign', data_dict, 'malignancy')

                # In case that Annotation_Content is missing, fill it by 'Missing_Annotation'
                else:
                    data_dict['nodule_type'] = 'Missing_Annotation'
                    data_dict['lobe'] = 'Missing_Annotation'
                    data_dict['pathology'] = 'Missing_Annotation'
                    data_dict['malignancy'] = 'Missing_Annotation'  

                # contours and sliceIdx
                contours_dict = {}
                for contours in data['Annotation_Contours']:
                    contours_dict[contours['imageOrder']] = contours['brush']
                    data_dict['Annotation_Contours'] = contours_dict


            group_dict[group] = data_dict
            
        series_dict[series_id] = group_dict
    #print(series_dict)
    return series_dict


# For new_grouped_nodule_id.json
def from_series_dict_to_df(json_dir, ct_series_dir, series_dict, dst):
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
        for group, group_infos in series_value.items():

            Group_ID = str(group)
            Patient_ID = group_infos['patient_id']
            Study_ID = group_infos['study_id']
            Annotation_ID = str(group_infos['annotation_id'])
            Annotation_Label = str(group_infos['annotation_label'])
            #Annotation_Lesion_ID = str(group_infos['annotation_lesion_id'])
            Annotation_Action = str(group_infos['annotation_action'])
            Ref_Annotation_ID = str(group_infos['ref_annotation_id'])
            Nodule_Type = group_infos['nodule_type']
            Lobe = group_infos['lobe']
            Pathology = group_infos['pathology']
            Malignancy = str(group_infos['malignancy'])

            imgOrder_list = []
            mask_list = []     
            for imageOrder, mask_info in group_infos['Annotation_Contours'].items():

                SliceIdx = imageOrder                
                Mask = mask_info['compressed']
                ## decode mask
                mask = decode_mask(Mask)
                mask_list.append(mask)
                imgOrder_list.append(SliceIdx)

                # save in slice-based information
                # basic_info_lst = [Patient_ID, Study_ID,Series_ID, Group_ID, Annotation_ID, Annotation_Action, Annotation_Label, Ref_Annotation_ID]
                # nodule_info_lst = [Nodule_Type, Lobe, Pathology, Malignancy, SliceIdx, Mask]
                # #nodule_info_lst = [Nodule_Type, Lobe, Pathology, Malignancy, mask]
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

            # print(volume)

            basic_info_lst = [Patient_ID, Study_ID,Series_ID, Group_ID, Annotation_ID, Annotation_Action, Annotation_Label, Ref_Annotation_ID]
            nodule_info_lst = [Nodule_Type, Lobe, Pathology, Malignancy, mask]
            column_value_lst = basic_info_lst + nodule_info_lst
            new_data.append(column_value_lst)

    new_df = pd.DataFrame(new_data, columns = ['Patient_ID', 'Study_ID','Series_ID', 'Group_ID','Annotation_ID', 'Annotation_Action', 
                                'Annotation_Label', 'Ref_Annotation_ID', 'Nodule_Type', 'Lobe', 'Pathology', 'Malignancy', 'Mask'])
    #new_df.to_csv(dst)
    print(new_df)

    return new_df


@nb.jit(nopython = True, cache = True)
def calculate_IoU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    overlay_area = np.sum(intersection) / np.sum(union)

    return overlay_area

def compare2df_match(predict_df, group_df):
    """
    Compare two dataframes (group df and predict df) to find Group ID in group df.

    Comparison steps: 
    First step: If series ID is the same, combine mask1 and mask2 to be a combination list.
    Second step: Iterating mask1 and mask2 in combination list to calculate the overlay area. 
    Third step:  If the overlay area of two masks are more than 0.0, check the series ID and mask 
                in group df is equal to the caculated mask and return the "Annotation Label" in group df.
    Forth step: Also, check the series ID and mask in predict df is equal to the other caculated mask or not.
                If so, the group id will be add in the "Group ID" column in predict df.
    """

    #predict_df = predict_df.sort_values(by = 'Volume')
    #group_df = group_df.sort_values(by = 'Volume')
    predict_df["Group_ID"] = ""


    for j, row in predict_df.iterrows():
        series_id = row['Series_ID']
        print(series_id)
        
        # If series ID is the same, iterate mask1 and mask2 combinations.
        mask1_list = predict_df.loc[predict_df['Series_ID'] == series_id, 'Mask'].tolist()
        mask2_list = group_df.loc[group_df['Series_ID'] == series_id, 'Mask'].tolist()
        combinations_list = [(x, y, i) for (_, x), (i, y) in itertools.product(enumerate(mask1_list), enumerate(mask2_list))]

        group_ids = []        
        # Calculate Overlay area and return the Group ID
        for i, (mask1, mask2, index) in enumerate(combinations_list):          
            # skip the None mask
            if np.any(mask1) is None or np.any(mask2) is None:
                continue
            else:
                #print(mask1.shape, mask2.shape)
                # calculate IoU
                overlay_area = calculate_IoU(mask1, mask2)
                print(f'overlay:', overlay_area)
            
            
            # If Overlay area > 0.1, matched Group ID will add to list
            if overlay_area > 0.0:
                filter = (group_df['Series_ID'] == series_id) &  (group_df['Mask'].apply(lambda x: np.array_equal(x, mask2)))
                group_id =  group_df.loc[filter, 'Annotation_Label'].values[0]
                
                predict_filter = (predict_df['Series_ID'] == series_id) &  (predict_df['Mask'].apply(lambda x: np.array_equal(x, mask1)))
                predict_df['Group_ID'].loc[predict_filter] = group_id

            else:
                group_id = None
    

    # In predict df add new column "Group ID"
    # According to the Overlay area to determine return the Group ID or not
    # axis = 1: apply function to each row
    #predict_df['Group_ID'] = predict_df.apply(calculate_overlay, axis=1)

    predict_df.to_csv('batch_12_match_0.0.csv')



if __name__=='__main__':

    """
    1. Covert new_grouped_nodule_id.json to group dataframe.
    2. Covert AI predict json file (balance.json) to predict dataframe.
    3. Compare and Match two dataframes.
    
    """
    
    # # For new_grouped_nodule_id.json
    print('Start dealing with group json', flush=True)
    json_path = "./res/batch_12_review_2/new_grouped_nodule_id_.json"
    json_dict = load_json(json_path)
    series_dict = return_nodule_info_dict(json_dict)
    #dst = os.path.join('./compare2json/', 'json_group_df_9_mask.csv')
    json_dir = "./annotations/batch_12_review_2/V20220927043003_Bt12_Ex3"
    ct_series_dir = "/mnt/data_4T/Elena/match2df/Datasets/batch_12/for_brenda/ct"
    group_df = from_series_dict_to_df(json_dir, ct_series_dir, series_dict, dst = None)

    
    ## for AI predict
    print('Start dealing with AI prediction', flush=True)
    json_path = "./predict_json/batch_12/json/"
    series_dict = return_nodule_info_dict_balance(json_path, batch_9=False)  
    #dst = os.path.join('./compare2json/', 'AI_predict_batch_12_mask.csv')
    json_dir = "./annotations/batch_12/V20220805111308_Bt12_Ex1"
    ct_series_dir = "/mnt/data_4T/Elena/match2df/Datasets/batch_12/for_brenda/ct"
    predict_df = from_series_dict_to_df_balance(json_dir, ct_series_dir,series_dict, dst = None) 


    ## compare two dataframes
    print('Start comparing', flush=True)
    compare2df_match(predict_df, group_df)



            
