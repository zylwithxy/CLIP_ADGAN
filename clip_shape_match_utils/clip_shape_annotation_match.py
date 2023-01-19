import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import clip
from tqdm import tqdm
from typing import List
from matplotlib import cm

# The input is the masked images, 
# for example, the clothes extracted by human parsing map and the text is long-sleeves, short-sleeves, sleeveless.

DATAROOT = '/media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion/fashion_resize'
DIR_SP = '/media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion'
PISE_SPL = '/media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion/trainSPL8'

def read_img(P1_name: str) -> Image.Image:
    
    dataroot, phase = DATAROOT ,'train'
    dir_path = os.path.join(dataroot, phase)
    assert os.path.exists(dir_path)
    input_P1_path = os.path.join(dir_path, P1_name)
    
    P1_img = Image.open(input_P1_path).convert('RGB')
    
    return P1_img


def read_mask(P1_name: str) -> np.ndarray:
    
    SP1_name = split_name(P1_name, 'semantic_merge3')
    dir_SP = DIR_SP
    SP1_path = os.path.join(dir_SP, SP1_name)
    SP1_path = SP1_path[:-4] + '.npy'
    assert os.path.exists(SP1_path)
    SP1_data = np.load(SP1_path)
    SP1 = np.zeros((8, 256, 176), dtype='uint8')
    for id in range(8):
        SP1[id] = (SP1_data == id).astype('uint8') # uint8
    
    return SP1


def show_part_img(imgs: list):
    
    fig, axes = plt.subplots(4, 2)
    axes = axes.flatten()
    
    for index, (axe, img) in enumerate(zip(axes, imgs)):
        axe.imshow(img)
        axe.set_title(f'{index}')
        axe.axis('off')
    
    plt.show()
    

def read_img_and_mask(input_P1_name: str, choice: str):
    """_summary_

    Args:
        input_P1_name (str): The filename of input person image.
        choice (str): 'masked_img' or 'original_img'

    Returns:
        _type_: _description_
    """
    # input_P1_name = 'fashionMENTees_Tanksid0000404701_7additional.jpg'
    P1_img = read_img(input_P1_name)
    if choice == 'original_img':
        # import pdb; pdb.set_trace()
        return P1_img
    P1_img = np.array(P1_img)
    P1_mask = read_mask(input_P1_name)
    imgs = add_mask_to_img(P1_mask, P1_img)
    
    # import pdb; pdb.set_trace()
    img_PIL = Image.fromarray(imgs[5]) # Which is clothes images and extracted by segmentation map.
    # plt.imshow(img_PIL)
    # plt.show()
    
    return img_PIL


def load_clip_models():
    
    torch.cuda.set_device(1)
    model, preprocess = clip.load("ViT-B/32", device= 'cuda')
    
    return model, preprocess


def clip_match(input_P1_name: str, model, preprocess, prompt: List[str], choice: str) -> np.ndarray:
    """_summary_

    Args:
        input_P1_name (str): The filename of input person image.
        model (_type_): CLIP pretrained model.
        preprocess (_type_): The preprocess of CLIP.
        prompt (List[str]): The text after prompt.
        choice (str): 'masked_img' or 'original_img'

    Returns:
        np.ndarray: _description_
    """
    
    img_PIL = read_img_and_mask(input_P1_name, choice)

    image = preprocess(img_PIL).unsqueeze(0).cuda()
    text = clip.tokenize(prompt).cuda()
    
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    # print("Label probs:", probs)
    return probs
    
    
def read_txt_file(choice:str = 'train') -> list:
    """Which is used to read the filename of train/test file.

    Args:
        choice (str, optional): 'train' or 'test'. Defaults to 'train'.
    """
    root_path = DIR_SP
    fname = choice + '.txt'
    train_test_path = os.path.join(root_path, fname)
    assert os.path.exists(train_test_path)
    
    fnames = []
    with open(train_test_path, 'r') as f:
        for row in f:
            row = row.rstrip('\n')
            fnames.append(row)
            
    return fnames


def write_similarity_files(choice_img: str, prompt, similarity_fnames: str = './clip_shape_annotations_similarity.txt', choice:str = 'train'):
    
    fnames = read_txt_file(choice)
    model, preprocess = load_clip_models()
    size = len(fnames)
    pbar = tqdm(total= size)
    
    with open(similarity_fnames, 'wt') as f:
        for index, fname in enumerate(fnames):
            simi_scores = clip_match(fname, model, preprocess, prompt, choice_img)
            simi_scores = np.around(simi_scores, 4)
            simi_scores_list = [f'{item:.4f}' for item in simi_scores[0].tolist()]
    
            if index == size - 1:
                f.write(' '.join(simi_scores_list))
            else:
                f.write(' '.join(simi_scores_list)+'\n')
                
            pbar.update(1)
    
     
def add_mask_to_img(mask: np.ndarray, img: Image.Image) -> list:
    imgs = []
    for i in range(mask.shape[0]):
        mask_map = np.repeat(np.expand_dims(mask[i, :, :], axis = -1), 3, axis= -1)
        imgs.append(mask_map * img)
    
    return imgs
    

def split_name(str,type):
    
    list = []
    list.append(type)
    if (str[len('fashion'):len('fashion') + 2] == 'WO'):
        lenSex = 5
    else:
        lenSex = 3
    list.append(str[len('fashion'):len('fashion') + lenSex])
    idx = str.rfind('id0')
    list.append(str[len('fashion') + len(list[1]):idx])
    id = str[idx:idx + 10]
    list.append(id[:2]+'_'+id[2:])
    pose = str[idx + 10:]
    list.append(pose[:4]+'_'+pose[4:])

    head = ''
    for path in list:
        head = os.path.join(head, path)
        
    return head


def test_effect():

    input_P1_name = 'fashionWOMENDressesid0000164002_2side.jpg'
    prompt = create_prompt()
    probs = clip_match(input_P1_name, *load_clip_models(), prompt, choice= 'masked_img')
    print("Label probs:", probs)


def create_prompt() -> List[str]:
    
    # Without prompt. accuracy: 0.3741
    # prompt = "The clothing is " accuracy: 0.0896
    # prompt = 'The sleeve length is ' 0.2180
    # prompt = 'The sleeve length of the upper clothing is ' 0.2921
    
    # For the full image
    # prompt = 'The sleeve length of the upper clothing is ' 0.2937
    # prompt = 'The sleeve length of the upper clothing of the person is '
    
    prompt = 'The sleeve length of the upper clothing of the person is '
    texts = ["sleeveless", "short-sleeve", "medium-sleeve", "long-sleeve", "not long-sleeve", "not visible"]
    
    prompt_texts = [prompt + text for text in texts]
    
    return prompt_texts


def test_mask_pos(input_P1_name: str, choice: str):
    """_summary_

    Args:
        input_P1_name (str): The filename of input person image.
        choice (str): 'masked_img' or 'original_img'

    Returns:
        _type_: _description_
    """
    # input_P1_name = 'fashionMENTees_Tanksid0000404701_7additional.jpg'
    P1_img = read_img(input_P1_name)
    if choice == 'original_img':
        # import pdb; pdb.set_trace()
        return P1_img
    
    P1_img = np.array(P1_img)
    P1_mask = read_mask(input_P1_name)
    imgs = add_mask_to_img(P1_mask, P1_img)
    
    show_part_img(imgs)
    

def process_seg_map(P1_name: str):
    """Replace the integer in segmentation map to meaningful RGB values.

    Args:
        P1_name (str): The filename of input person image.
    """
    
    SP1_name = split_name(P1_name, 'semantic_merge3')
    dir_SP = DIR_SP
    SP1_path = os.path.join(dir_SP, SP1_name)
    SP1_path = SP1_path[:-4] + '.npy'
    assert os.path.exists(SP1_path)
    SP1_data = np.load(SP1_path)
    
    SP1_data_visual = np.zeros((*SP1_data.shape, 4), dtype= np.float64)
    colours = cm.get_cmap('plasma', 8)
    cmap = colours(np.linspace(0, 1, 8))
    
    # import pdb; pdb.set_trace()
    
    for index in range(8):
        if index == 0:
            SP1_data_visual[SP1_data == index, :] = np.float64([0, 0, 0, 1])
        else:   
            SP1_data_visual[SP1_data == index, :] = cmap[index]
    
    plt.imshow(SP1_data_visual)
    plt.axis('off')
    plt.show()


def filter_seg_map(P1_name: str):
    """Filter the segmentation map without sleeves and arms.
    
    """
    SP1_name = split_name(P1_name, 'semantic_merge3')
    dir_SP = DIR_SP
    SP1_path = os.path.join(dir_SP, SP1_name)
    SP1_path = SP1_path[:-4] + '.npy'
    assert os.path.exists(SP1_path)
    SP1_data = np.load(SP1_path)
    
    SP1_data_visual = np.zeros((*SP1_data.shape, 4), dtype= np.float64)
    colours = cm.get_cmap('plasma', 8)
    cmap = colours(np.linspace(0, 1, 8))
    
    # import pdb; pdb.set_trace()
    
    for index in range(8):
        if index == 0 or index == 5 or index == 6:
            SP1_data_visual[SP1_data == index, :] = np.float64([0, 0, 0, 1]) # Remove the sleeves, arms and add black background.
        else:   
            SP1_data_visual[SP1_data == index, :] = cmap[index]
    
    plt.imshow(SP1_data_visual)
    plt.axis('off')
    plt.show()


def read_spl(input_P1_name: str) -> np.ndarray:
    
    root_path = PISE_SPL
    input_P1_name = input_P1_name[:-4] + '.png'
    fpath = os.path.join(root_path, input_P1_name)
    assert os.path.exists(fpath)
    
    regions = (40,0,216,256)
    SPL1_img = Image.open(fpath).crop(regions)
    print(SPL1_img.size)
    SPL1_img = np.array(SPL1_img) # max is 7, min is 0.
    
    return SPL1_img

def read_pise_mask(SPL1_img: np.ndarray):
    
    SP1 = np.zeros((8, 256, 176), dtype='uint8')
    for index in range(8):
        SP1[index] = (SPL1_img == index).astype('uint8') # uint8
    
    return SP1

def test_pise_SPL(input_P1_name: str):
    
    SPL1_img = read_spl(input_P1_name) # uint8
    P1_mask = read_pise_mask(SPL1_img)
    
    P1_img = np.array(read_img(input_P1_name)) # input P1 image
    
    # imgs = add_mask_to_img(P1_mask, P1_img)
    # show_part_img(imgs) # The index are 3 and 6 for sleeves, arms.
    
    SP1_data_visual = np.zeros((*SPL1_img.shape, 4), dtype= np.float64)
    colours = cm.get_cmap('plasma', 8)
    cmap = colours(np.linspace(0, 1, 8))
    
    import pdb; pdb.set_trace()
    
    SPL1_img[SPL1_img == 3] = 0
    SPL1_img[SPL1_img == 6] = 0
    
    for index in range(8):
        # if index == 0 or index == 3 or index == 6:
        if index == 0:
            SP1_data_visual[SPL1_img == index, :] = np.float64([0, 0, 0, 1]) # Remove the sleeves, arms and add black background.
        else:   
            SP1_data_visual[SPL1_img == index, :] = cmap[index]
    
    plt.imshow(SP1_data_visual)
    plt.axis('off')
    plt.show()
    

# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, need_dec=False, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
        
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if need_dec:
#        image_numpy = torch.argmax(image_numpy, 2)
        image_numpy = decode_labels(image_numpy.astype(int))
    else:
        image_numpy = (image_numpy + 1) / 2.0 *  bytes

    return image_numpy.astype(imtype)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    
    label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]
    
    h, w, c = mask.shape
    #assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros(( h, w, 3), dtype=np.uint8)

    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    tmp = []
    tmp1 = []
    for j_, j in enumerate(mask[:, :, 0]):
        for k_, k in enumerate(j):
            #tmp1.append(k)
            #tmp.append(k)
            if k < num_classes:
                pixels[k_,j_] = label_colours[k]
    #np.save('tmp1.npy', tmp1)
    #np.save('tmp.npy',tmp)
    outputs = np.array(img)
    #print(outputs[144,:,0])
    return outputs

    
def deliver_pise_clip_spl2(input_P1_name: str) -> torch.Tensor:
    """To validate the meaning of SPL2 after a series of processing steps.
    The process is the same as PISE.data.base_dataset.BaseDataset.__getitem__()
    """
    SPL1_img = read_spl(input_P1_name) # ndarray
    s1np = np.expand_dims(SPL1_img,-1)
    s1np = np.concatenate([s1np,s1np,s1np], -1)
    SPL1_img_pil = Image.fromarray(np.uint8(s1np))
    SPL1_img = np.expand_dims(np.array(SPL1_img_pil)[:,:,0],0)
    
    SPL2 = torch.from_numpy(SPL1_img).long()
    
    # max_num, min_num = torch.max(SPL2), torch.min(SPL2) # 7, 0
    
    return SPL2

def create_8_channels_segmentation_map(input_P1_name: str) -> torch.Tensor:
    
    
    tmp = deliver_pise_clip_spl2(input_P1_name)
    tmp = tmp.view(-1)
    h, w, num_class = 256, 176, 8
    
    ones = torch.sparse.torch.eye(num_class)
    ones = ones.index_select(0, tmp)
    
    SPL2_onehot = ones.view([h, w, num_class])
    SPL2_onehot = SPL2_onehot.permute(2,0,1)
    
    return SPL2_onehot


def convert_segmentation_map(input_P1_name: str):
    """Convert 8 channels segmentation map to 1 channel map (which is like)
    """
    SPL2_onehot = create_8_channels_segmentation_map(input_P1_name) # shape (num_class, h, w)
    labels_again = torch.argmax(SPL2_onehot, dim=0).unsqueeze(0)
    
    image_numpy = tensor2im(labels_again, need_dec= True) # ndarray
    
    # import pdb; pdb.set_trace()
    
    plt.imshow(image_numpy)
    plt.xlabel('seg map', fontsize= 15)
    plt.show()
    
    # return labels_again


def show_pise_clip_spl2(input_P1_name: str):
    """Show the full segmentation map and masked segmentation map

    Args:
        input_P1_name (str): The filename of input person image.
    """
    
    SPL2_tensor = deliver_pise_clip_spl2(input_P1_name) # torch.Tensor
    
    SPL2_tensor_mask = torch.zeros_like(SPL2_tensor,dtype= SPL2_tensor.dtype) # Create mask
    SPL2_tensor_mask.copy_(SPL2_tensor.data)
    
    SPL2_tensor_mask[SPL2_tensor_mask == 3] = 0
    SPL2_tensor_mask[SPL2_tensor_mask == 6] = 0
    
    image_numpy = tensor2im(SPL2_tensor, need_dec= True) # The full segmentation map
    image_numpy_mask = tensor2im(SPL2_tensor_mask, need_dec= True)
    
    # import pdb; pdb.set_trace()
    images = [image_numpy_mask, image_numpy]
    
    fig, axes = plt.subplots(1, 2, tight_layout= True)
    axes = axes.flatten()
    titles = ['masked segmentation map', 'full segmentation map']
    
    for axe, img, title in zip(axes, images, titles):
        axe.imshow(img)
        axe.set_title(title)
        axe.axis('off')
    
    plt.show()
 

def main():
    
    # similarity_file_named_rules: './clip_shape_annotations_similarity_prompt{prompt_num}.txt'
    choice_img = 'original_img'
    prompt_num = 4  
    similarity_file_name = f'./clip_shape_annotations_similarity_prompt{prompt_num}_{choice_img}.txt'
    
    prompt_texts = create_prompt()
    
    # import pdb; pdb.set_trace()
    
    if not os.path.exists(similarity_file_name):
        write_similarity_files(choice_img, prompt_texts, similarity_file_name, 'train')
        
    else:
        print('The file exists !')
    

if __name__ == "__main__":
    
    # main()
    input_P1_name = 'fashionMENTees_Tanksid0000404701_7additional.jpg'
    choice = 'masked_img'
    
    # test_mask_pos(input_P1_name, choice)
    # process_seg_map(input_P1_name)
    # filter_seg_map(input_P1_name)
    # test_pise_SPL(input_P1_name)
    # show_pise_clip_spl2(input_P1_name)
    convert_segmentation_map(input_P1_name)