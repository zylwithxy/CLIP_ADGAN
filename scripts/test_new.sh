NAME="checkpoints_gpu12_CLIP_13_text_PCA_DALLE2"
export CHECK_DIR="./$NAME"
export RESULTS_DIR="./results_${NAME}"

python test.py --dataroot /media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion/fashion_resize \
               --dirSem /media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion \
               --pairLst /media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion/deepmultimodal-resize-pairs-test.csv \
               --checkpoints_dir $CHECK_DIR \
               --results_dir $RESULTS_DIR \
               --name DALLE2_img_text_PCA \
               --model adgan \
               --phase test \
               --dataset_mode keypoint \
               --norm instance \
               --batchSize 1 \
               --resize_or_crop no \
               --gpu_ids 2,1 \
               --BP_input_nc 18 \
               --no_flip \
               --which_model_netG ADGen \
               --which_epoch 100 \
               --choice_txt_img False \
               --use_PCA True \
               --prior_type MLP \
               --display_id 0 \
               --use_CLIP_img_txt_loss True