@echo off
REM CSCA增强的CycleGAN训练脚本 (Windows版本)
REM 使用方法：train_csca.bat [数据集名称]

set DATASET=%1
if "%DATASET%"=="" set DATASET=horse2zebra

set DATAROOT=./datasets/%DATASET%
set GPU_IDS=0
set BATCH_SIZE=1
set NITER=25
set NITER_DECAY=25

echo === 训练CSCA增强的CycleGAN模型 ===
echo 数据集: %DATASET%
echo 数据路径: %DATAROOT%

python train.py --dataroot %DATAROOT% --name %DATASET%_csca --model cycle_gan --netG resnet_9blocks --pool_size 50 --no_dropout --gpu_ids %GPU_IDS% --batch_size %BATCH_SIZE% --niter %NITER% --niter_decay %NITER_DECAY% --display_freq 400 --display_id 1 --display_server "http://localhost" --display_port 8097 --print_freq 100 --save_epoch_freq 5 --sn_gan 1 --no_lsgan --wgan 0 --with_gp 0 --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 0.5 --csca_training_stage full --csca_mid_init_gate 0.05 --csca_final_init_gate 0.1 --csca_reduction 4 --csca_cot_k 3 --csca_l1_weight 1e-4 --lambda_csca 0.001

echo 训练完成！
echo 模型保存在: ./checkpoints/%DATASET%_csca/
pause
