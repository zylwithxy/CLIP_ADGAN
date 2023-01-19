import os
from typing import List, Dict
from tqdm import tqdm
from clip_shape_annotation_match import read_txt_file
from predict_shape_accuracy import read_all_shape0
import pandas as pd

ROOT = '/media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion'

def write_shape0_file(choice: str):
    """Write the shape0 file for train/test file. The format is as followed:
    
       filename                 shape (The split symbol is space)
       fashionWOMENTees_Tanksid0000570402_7additional.jpg   3 (The range is 0-5)
       
    Args:
        choice (str): 'train' or 'test'.
    """
    assert choice == 'train' or choice == 'test'
    
    fnames: List[str] = read_txt_file(choice) # list
    fname_shape_attribute = read_all_shape0()
    
    train_test_shape_values: List[str] = [fname_shape_attribute[fname] for fname in fnames]
    
    save_root = ROOT
    file_name = f'shape_{choice}.txt'
    save_path = os.path.join(save_root, file_name)
    
    size = len(fnames)
    pbar = tqdm(total= size)
    
    with open(save_path, 'wt') as f:
        for index, (fname, value) in enumerate(zip(fnames, train_test_shape_values)):
            if index == size - 1:
                f.write(fname + ' ' + value)
            else:
                f.write(fname + ' ' + value + '\n')
            
            pbar.update(1)
    
    print("Successful writing!")


def test_retrieve_test_file(choice: str):
    """
    Test read shape0 file by the way in the __getitem__ method of BaseDataset
    
    Args:
        choice (str): 'train' or 'test'.
    """
    
    root = ROOT
    fname = f'deepmultimodal-resize-pairs-{choice}.csv'
    pairLst = os.path.join(root, fname)
    
    def init_categories(pairLst) -> List[List[str]]:
        assert os.path.exists(pairLst)
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')  
        return pairs
    
    def read_shape0_file(choice: str) -> Dict[str, int]:
        shape_path = os.path.join(ROOT, f'shape_{choice}.txt')
        assert os.path.join(shape_path)
        
        fname_shape_pair = dict()
        with open(shape_path, 'rt') as f:
            for row in f:
                row_list = row.split()
                fname_shape_pair[row_list[0]] = int(row_list[1])
        
        return fname_shape_pair
        
    pairs = init_categories(pairLst)
    fname_shape_pair = read_shape0_file(choice)
    P1_name, P2_name = pairs[0]
    
    # Retrieve the text
    texts = ["sleeveless", "short-sleeve", "medium-sleeve", "long-sleeve", "not long-sleeve", "not visible"]
    index = fname_shape_pair[P1_name]
    import pdb; pdb.set_trace()
    # texts[index]
    

if __name__ == "__main__":
    
    choice = 'train'
    # write_shape0_file(choice)
    test_retrieve_test_file(choice)