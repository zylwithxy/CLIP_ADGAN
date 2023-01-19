import os
from typing import List, Dict
from clip_shape_annotation_match import read_txt_file
import torch

DIR_SHAPE_ALL = '/media/beast/WD2T/XUEYu/dataset_pose_transfer/DeepFashionMultiModal/DeepFashionMultiModal_original/labels/shape'

def process_symbol(temp: str):
    ids = []
    for id, item in enumerate(temp):
        if item == '_':
            ids.append(id)
    
    return ids


def process_splitkey(captions_list: List[str]) -> List:
    """Remove '_' in each item in captions_list

    Args:
        captions_list (List[str]): ['Sex', 'Clothing kinds', 'id', '01_7_additional']
    """
    assert len(captions_list) == 4
    captions_replace = []
    
    for id, item in enumerate(captions_list):
        if id == 2:
            item_split = item.split('_') # list
            item_concat = ''.join(item_split)
            captions_replace.append(item_concat)
        elif id == 3:
            ids = process_symbol(item)
            item_concat = item[:ids[-1]] + item[ids[-1]+1: ]
            captions_replace.append(item_concat)
        else:
            captions_replace.append(item)
    
    return captions_replace


def change_captions_keys(captions_keys: List[str]) -> List[str]:
    """Which is used to change the keys of captions.
    For example, 'MEN-Jackets_Vests-id_00003336-10_3_back' to 'fashionMENJacketsVestsid0000333610_3back.jpg'
    """
    captions_list = []
    for key in captions_keys:
        key_split = key.split('-') # ['Sex', 'Clothing kinds', 'id', '01_7_additional']
        key_replace = process_splitkey(key_split)
        temp_final = ''.join(key_replace)
        captions_list.append('fashion' + temp_final + '.jpg')
    
    return captions_list


def read_all_shape0() -> Dict[str, str]:
    
    fname = 'shape_anno_all.txt'
    root_shape = DIR_SHAPE_ALL
    shape_path = os.path.join(root_shape, fname)
    assert os.path.exists(shape_path)

    keys, values = [], []
    with open(shape_path, 'r') as f:
        for row in f:
            annotations = row.split()
            key, value = annotations[0], annotations[1] # file name; shape[0] sleeve length.
            keys.append(os.path.splitext(key)[0])
            values.append(value)
    
    keys = change_captions_keys(keys)
    fname_shape_attribute = dict(zip(keys, values)) # len is 42544
    
    return fname_shape_attribute


def retrieve_train_values() -> List[int]:
    
    fnames = read_txt_file('train')
    fname_shape_attribute = read_all_shape0()
    
    train_shape_values = [int(fname_shape_attribute[fname]) for fname in fnames]
    
    return train_shape_values


def retrieve_clip_similarity(clip_similarity_fnames: str) -> torch.Tensor:
    
    train_shape_values_predict = []
    assert os.path.exists(clip_similarity_fnames)
    with open(clip_similarity_fnames, 'rt') as f:
        for row in f:
            scores_row = row.split()
            scores_row = torch.tensor([float(score) for score in scores_row])
            train_shape_values_predict.append(scores_row.argmax())
    
    train_shape_values_predict = torch.tensor(train_shape_values_predict)
    
    return train_shape_values_predict


def convert_not_long_sleeve(choice: int, input_tensor: torch.Tensor):
    """Convert the not long-sleeve to other types.

    Args:
        choice (int): 0 means "sleeveless", 1 means "short-sleeve", 2 means "medium-sleeve"
    """
    count = 0
   
    for id, item in enumerate(input_tensor):
        if item == 4: # "not long-sleeve"
            if choice == 0: 
                input_tensor[id] = 0
            elif choice == 1:
                input_tensor[id] = 1
            elif choice == 2:
                input_tensor[id] = 2
            count += 1
    
    print(f'The count is {count}')


def cal_predict_accuracy(clip_similarity_fnames: str, enable_convert: bool):
    
    train_shape_values_true = retrieve_train_values()
    train_shape_values_true = torch.tensor(train_shape_values_true) # torch.tensor
    train_shape_values_predict = retrieve_clip_similarity(clip_similarity_fnames)
    
    if enable_convert:
        convert_not_long_sleeve(3, train_shape_values_true)
        convert_not_long_sleeve(3, train_shape_values_predict)
    
    assert torch.all(train_shape_values_true <= torch.tensor(5).type(train_shape_values_true.dtype))
    assert torch.all(train_shape_values_predict <= torch.tensor(5).type(train_shape_values_true.dtype))
    
    assert torch.all(train_shape_values_true >= torch.tensor(0).type(train_shape_values_true.dtype))
    assert torch.all(train_shape_values_predict >= torch.tensor(0).type(train_shape_values_predict.dtype))
    
    accuracy = torch.sum(train_shape_values_true == train_shape_values_predict) / len(train_shape_values_true)
    print(f'The accuracy is {accuracy: .4f}')
    

if __name__ == "__main__":
    
    # similarity_file_named_rules: './clip_shape_annotations_similarity.txt'
    
    choice_img = 'original_img' # If the inputP1 image is the full image.
    
    prompt_num = 4
    # similarity_file_name = f'./clip_shape_annotations_similarity_prompt{prompt_num}.txt'
    # similarity_file_name = './clip_shape_annotations_similarity.txt'
    similarity_file_name = f'./clip_shape_annotations_similarity_prompt{prompt_num}_{choice_img}.txt'
    cal_predict_accuracy(similarity_file_name, enable_convert= True)
    
    pass