#!/bin/bash
epoch=474
dataset='Flow'
while (( $epoch<475 ))
do
	python eval_derain.py --gpus 4 --model 'weights/4x_ubunRBPN_backbone_de_eq_attn_3_epoch_'$epoch'.pth'
	python eval_psnr_ssim.py --epoch $epoch --dataset $dataset 
	let "epoch=epoch+25"
done

