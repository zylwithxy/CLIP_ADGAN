import numpy as np
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# losses
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss
from losses.CX_style_loss import CXLoss
from .vgg import VGG

import sys
sys.path.append('..')
import clip

class TransferModel(BaseModel):
    def name(self):
        return 'TransferModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize # 256
        self.input_P1_set = self.Tensor(nb, opt.P_input_nc, size, size) # (2, 3, 256, 256)
        self.input_BP1_set = self.Tensor(nb, opt.BP_input_nc, size, size) # (2, 18, 256, 256)
        self.input_P2_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BP2_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        # self.input_SP1_set = self.Tensor(nb, opt.SP_input_nc, size, size) # (2, 8, 256, 256)

        input_nc = [opt.P_input_nc, opt.BP_input_nc+opt.BP_input_nc]
        self.netG = networks.define_G(input_nc, opt.P_input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        n_downsampling=opt.G_n_downsampling, use_PCA= opt.use_PCA, opt_prior_type= opt.prior_type, opt_choice_txt_img= opt.choice_txt_img)
        self.choice = opt.choice_txt_img
        self.use_PCA = opt.use_PCA
        self.use_CLIP_img_txt_loss = opt.use_CLIP_img_txt_loss

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.with_D_PB:
                self.netD_PB = networks.define_D(opt.P_input_nc+opt.BP_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(opt.P_input_nc+opt.P_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if len(opt.gpu_ids) > 1:
                self.load_VGG(self.netG.module.enc_style.vgg)
            else:
                self.load_VGG(self.netG.enc_style.vgg)





        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            if opt.L1_type == 'origin':
                self.criterionL1 = torch.nn.L1Loss()
            elif opt.L1_type == 'l1_plus_perL1':
                self.criterionL1 = L1_plus_perceptualLoss(opt.lambda_A, opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)
            else:
                raise Exception('Unsurportted type of L1!')

            if opt.use_cxloss:
                self.CX_loss = CXLoss(sigma=0.5)
                if torch.cuda.is_available():
                    self.CX_loss.cuda()
                self.vgg = VGG()
                self.vgg.load_state_dict(torch.load(os.path.abspath(os.path.dirname(opt.dataroot)) + '/vgg_conv.pth'))
                for param in self.vgg.parameters():
                    param.requires_grad = False
                if torch.cuda.is_available():
                    self.vgg.cuda()

            self.clip_img_txt_loss = nn.MSELoss(reduction='mean') # define loss functions for CLIP image embeddings and text embeddings.
            
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))


            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            if opt.with_D_PB:
                networks.print_network(self.netD_PB)
            if opt.with_D_PP:
                networks.print_network(self.netD_PP)
        print('-----------------------------------------------')
        
        self.attr_embedder, _ = clip.load("ViT-B/32", device= 'cuda' if self.gpu_ids else 'cpu')

    def set_input(self, input:dict):
        """_summary_

        Args:
            input (dict): Data in the keypoint dataset.
            choice(bool): True means using the CLIP image encoder. False means use the CLIP text encoder.
            use_PCA(bool): True means using PCA to reduce the dimension especially for the 13 attributes of CLIP text embeddings.
        """
        input_P1, input_BP1 = input['P1'], input['BP1']
        input_P2, input_BP2 = input['P2'], input['BP2']

        self.input_P1_set.resize_(input_P1.size()).copy_(input_P1)
        self.input_BP1_set.resize_(input_BP1.size()).copy_(input_BP1)
        self.input_P2_set.resize_(input_P2.size()).copy_(input_P2)
        self.input_BP2_set.resize_(input_BP2.size()).copy_(input_BP2)

        #input_SP1 = input['SP1']
        # self.input_SP1_set.resize_(input_SP1.size()).copy_(input_SP1)
        if not self.choice:
            self.input_TXT1: torch.Tensor = input['TXT1'] if not self.use_PCA else input['TXT1'].float() # if not self.use_PCA; tensors. 13 attributes
        
        if self.choice or self.use_CLIP_img_txt_loss:
            self.input_IMG1 = input['CLIP_img_input'].cuda() if self.gpu_ids else input['CLIP_img_input']
        
        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]
        self.person_paths = input['P1_path'][0]

    def forward(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)
        
        # self.input_SP1 = Variable(self.input_SP1_set)
        if not self.choice:
            # text = clip.tokenize(self.input_TXT1).cuda()
            # input_TXT1 = self.attr_embedder.encode_text(text)
            if not self.use_PCA:
                style_code = self.input_TXT1.float().to(torch.device(f'cuda:{self.gpu_ids[0]}'))
            else:
                # assert self.input_TXT1.__class__.__name__ == 'ndarray'
                style_code = self.input_TXT1.cuda() # shape torch.Size([b, 512])
            
            if self.use_CLIP_img_txt_loss:
                style_img = self.attr_embedder.encode_image(self.input_IMG1)
                style_img = style_img.float()
            
        else:
            style_code = self.attr_embedder.encode_image(self.input_IMG1)
            style_code = style_code.float()

        self.fake_p2, self.style_prior = self.netG(self.input_BP2, self.input_P1, style_code)



    def test(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)

        # self.input_SP1 = Variable(self.input_SP1_set)
        if self.choice:
            style_code = self.attr_embedder.encode_image(self.input_IMG1).float()
        else:
            style_code = self.input_TXT1.float().cuda()
        
        self.fake_p2, _ = self.netG(self.input_BP2, self.input_P1, style_code)


    # get image paths
    def get_image_paths(self) -> str:
        return self.image_paths

    def get_person_paths(self):
        return self.person_paths


    def backward_G(self):
        if self.opt.with_D_PB:
            pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), 1))
            self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), 1))
            self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)

        # CX loss
        if self.opt.use_cxloss:
            style_layer = ['r32', 'r42']
            vgg_style = self.vgg(self.input_P2, style_layer)
            vgg_fake = self.vgg(self.fake_p2, style_layer)
            cx_style_loss = 0

            for i, val in enumerate(vgg_fake):
                cx_style_loss += self.CX_loss(vgg_style[i], vgg_fake[i])
            cx_style_loss *= self.opt.lambda_cx

        pair_cxloss = cx_style_loss
        
        if self.use_CLIP_img_txt_loss:
            clip_img_txt_loss = self.clip_img_txt_loss(self.style_prior, self.attr_embedder.encode_image(self.input_IMG1).float())
            clip_img_txt_loss *= self.opt.lambda_CLIP

        # L1 loss
        if self.opt.L1_type == 'l1_plus_perL1' :
            losses = self.criterionL1(self.fake_p2, self.input_P2)
            self.loss_G_L1 = losses[0]
            self.loss_originL1 = losses[1].data
            self.loss_perceptual = losses[2].data

        else:
            self.loss_G_L1 = self.criterionL1(self.fake_p2, self.input_P2) * self.opt.lambda_A

        pair_L1loss = self.loss_G_L1

        if self.opt.with_D_PB:
            pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_GAN
            if self.opt.with_D_PP:
                pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_GAN
                pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_GAN


        if self.opt.with_D_PB or self.opt.with_D_PP:
            pair_loss = pair_L1loss + pair_GANloss
        else:
            pair_loss = pair_L1loss

        if self.opt.use_cxloss:
            pair_loss = pair_loss + pair_cxloss
            
        if self.use_CLIP_img_txt_loss:
            pair_loss += clip_img_txt_loss

        pair_loss.backward()

        self.pair_L1loss = pair_L1loss.data
        if self.opt.with_D_PB or self.opt.with_D_PP:
            self.pair_GANloss = pair_GANloss.data

        if self.opt.use_cxloss:
            self.pair_cxloss = pair_cxloss.data
        
        if self.use_CLIP_img_txt_loss:
            self.clip_loss_result = clip_img_txt_loss.data


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self):
        real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
        # fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
        fake_PB = self.fake_PB_pool.query( torch.cat((self.fake_p2, self.input_BP2), 1).data )
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB)

        self.loss_D_PB = loss_D_PB.data

    # D: take(P, P') as input
    def backward_D_PP(self):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        # fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
        fake_PP = self.fake_PP_pool.query( torch.cat((self.fake_p2, self.input_P1), 1).data )
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP)

        self.loss_D_PP = loss_D_PP.data


    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_P
        if self.opt.with_D_PP:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PP.zero_grad()
                self.backward_D_PP()
                self.optimizer_D_PP.step()

        # D_BP
        if self.opt.with_D_PB:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PB.zero_grad()
                self.backward_D_PB()
                self.optimizer_D_PB.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([ ('pair_L1loss', self.pair_L1loss)])
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB'] = self.loss_D_PB
        if self.opt.with_D_PB or self.opt.with_D_PP or self.opt.with_D_PS:
            ret_errors['pair_GANloss'] = self.pair_GANloss

        if self.opt.L1_type == 'l1_plus_perL1':
            ret_errors['origin_L1'] = self.loss_originL1
            ret_errors['perceptual'] = self.loss_perceptual

        if self.opt.use_cxloss:
            ret_errors['CXLoss'] = self.pair_cxloss
        
        if self.use_CLIP_img_txt_loss:
            ret_errors['CLIP_img_txt_Loss'] = self.clip_loss_result
            
        return ret_errors

    def get_current_visuals(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)

        input_BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]

        fake_p2 = util.tensor2im(self.fake_p2.data)

        vis = np.ones((height, width*5, 3)).astype(np.uint8) #h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width*2, :] = input_BP1
        vis[:, width*2:width*3, :] = input_P2
        vis[:, width*3:width*4, :] = input_BP2
        vis[:, width*4:, :] = fake_p2

        # temp = os.path.splitext(self.person_paths)[0]
        # key = temp[7:]
        key = self.get_image_paths()
        ret_visuals = OrderedDict([(key, vis)])
        
        text = 'Image embeddings' if self.choice else '13 Attributes'

        return ret_visuals, text # [0] is due to the tensor2im choose the batch[0].


    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB,  'netD_PB',  label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)


