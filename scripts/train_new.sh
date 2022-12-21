python train.py --dataroot /media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion/fashion_resize \
                --dirSem /media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion \
                --pairLst /media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion/deepmultimodal-resize-pairs-train.csv \
                --name fashion_adgan_test \
                --model adgan \
                --nThreads 0 \
                --lambda_GAN 5 \
                --lambda_A 1 \
                --lambda_B 1 \
                --dataset_mode keypoint \
                --n_layers 3 \
                --norm instance \
                --batchSize 12 \
                --pool_size 0 \
                --resize_or_crop no \
                --gpu_ids 0,1,2 \
                --BP_input_nc 18 \
                --SP_input_nc 8 \
                --no_flip \
                --which_model_netG ADGen \
                --niter 50 \
                --niter_decay 50 \
                --checkpoints_dir ./checkpoints_gpu012_new \
                --L1_type l1_plus_perL1 \
                --n_layers_D 3 \
                --with_D_PP 1 \
                --with_D_PB 1 \
                --display_id 0 \
                --img_dir /media/beast/WD2T/XUEYu/processed_dataset/train_images \
                --pose_dir /media/beast/WD2T/XUEYu/processed_dataset/densepose \
                --segm_dir /media/beast/WD2T/XUEYu/processed_dataset/segm \