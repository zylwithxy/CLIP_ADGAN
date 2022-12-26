import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch

# import sys
# sys.path.append(os.path.expanduser('~/XUEYu/pose_transfer/CLIP_ADGAN'))
# sys.path.append(os.path.expanduser('~/XUEYu/pose_transfer/CLIP_ADGAN/ADGAN'))
# from read_captions import Read_MultiModel_Captions

import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class KeyDataset(BaseDataset):
    
    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def clip_transform(self, n_px):
        return Compose([
            Resize(n_px, Image.BICUBIC),
            CenterCrop(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


    def _open_file(self, path_prefix, fname):
        return open(os.path.join(path_prefix, fname), 'rb')
    
    def _load_densepose(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        fname = f'{fname[:-4]}_densepose.png'
        with self._open_file(self._densepose_path, fname) as f:
            densepose = Image.open(f)
            if self.downsample_factor != 1:
                width, height = densepose.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                densepose = densepose.resize(
                    size=(width, height), resample=Image.NEAREST)
            # channel-wise IUV order, [3, H, W]
            densepose = np.array(densepose)[:, :, 2:].transpose(2, 0, 1)
        return densepose.astype(np.float32)
    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase) #person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K') # Densepoints
        self.dir_TEXT = os.path.join(opt.dataroot, opt.phase + '_text') # 13 CLIP text embeddings
        self.dir_SP = opt.dirSem #semantic deepfashion path
        self.SP_input_nc = opt.SP_input_nc
        self.gpu_ids = opt.gpu_ids
        self.choice_txt_img = opt.choice_txt_img
        
        # text_caption = Read_MultiModel_Captions()
        # self.text_cap: dict[str, str] = text_caption.captions_adgan

        self.init_categories(opt.pairLst) # deepmultimodal-resize-pairs-train.csv
        self.transform = get_transform(opt)
        self.clip_tran = self.clip_transform(224)
        # _, preprocess = clip.load("ViT-B/32", device= 'cuda' if self.gpu_ids else 'cpu')
        # self.process = preprocess
        

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)

        P1_name, P2_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name) # person 1
        BP1_path = os.path.join(self.dir_K, P1_name + '.npy') # bone of person 1

        # person 2 and its bone
        P2_path = os.path.join(self.dir_P, P2_name) # person 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy') # bone of person 2


        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = np.load(BP1_path) # h, w, c
        BP2_img = np.load(BP2_path) 
        # use flip
        if self.opt.phase == 'train' and self.opt.use_flip: # use_flip == 0
            # print ('use_flip ...')
            flip_random = random.uniform(0,1)
            
            if flip_random > 0.5:
                # print('fliped ...')
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT) # 镜像

                BP1_img = np.array(BP1_img[:, ::-1, :]) # flip
                BP2_img = np.array(BP2_img[:, ::-1, :]) # flip

            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        else:
            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        # segmentation ; we don't use the segmentation map.
        """
        SP1_name = self.split_name(P1_name, 'semantic_merge3') # 按位置分割选中 semantic_merge3 对应的pic
        SP1_path = os.path.join(self.dir_SP, SP1_name)
        SP1_path = SP1_path[:-4] + '.npy' # 原来的后缀是.jpg
        SP1_data = np.load(SP1_path)
        SP1 = np.zeros((self.SP_input_nc, 256, 176), dtype='float32') # 8通道, 每个通道对应的binary属性. e.g. head, clothes, armes etc.
        for id in range(self.SP_input_nc):
            SP1[id] = (SP1_data == id).astype('float32')
        """
        # TXT1 = self.text_cap[P1_name]
        TXT_13 = os.path.join(self.dir_TEXT, os.path.splitext(P1_name)[0] + '.pt')
        Txt_Embeddings = torch.load(TXT_13)
        
        
        dict_return = {'P1': P1, 'BP1': BP1, 'TXT1': Txt_Embeddings, 'P2': P2, 'BP2': BP2,
                'P1_path': P1_name, 'P2_path': P2_name, 'CLIP_img_input': self.clip_tran(Image.open(P1_path))} if self.choice_txt_img else \
                {'P1': P1, 'BP1': BP1, 'TXT1': Txt_Embeddings, 'P2': P2, 'BP2': BP2,
                'P1_path': P1_name, 'P2_path': P2_name}
        
        return dict_return # Test the CLIP image encoder effect

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'

    def split_name(self,str,type):
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

