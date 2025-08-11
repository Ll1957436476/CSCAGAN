#!/bin/bash

# CSCA增强的CycleGAN训练脚本
# 使用方法：bash scripts/train_cyclegan_csca.sh [数据集名称]

set -ex

# 配置参数
DATASET=${1:-"horse2zebra"}  # 默认使用horse2zebra数据集
DATAROOT="./datasets/${DATASET}"
GPU_IDS="0"
BATCH_SIZE=1
NITER=25                 # 固定学习率训练25个epoch
NITER_DECAY=25           # 学习率衰减训练25个epoch (总共50个epoch)

echo "=== 训练CSCA增强的CycleGAN模型 ==="
echo "数据集: ${DATASET}"
echo "数据路径: ${DATAROOT}"

python train.py \
    --dataroot $DATAROOT \                      # 数据集路径
    --name ${DATASET}_csca \                    # 实验名称，用于保存模型和结果
    --model cycle_gan \                         # 使用CycleGAN模型
    --netG resnet_6blocks \                     # 生成器架构：6层ResNet块
    --pool_size 50 \                           # 图像池大小，存储之前生成的图像
    --no_dropout \                             # 不使用dropout
    --gpu_ids $GPU_IDS \                       # GPU设备ID
    --batch_size $BATCH_SIZE \                 # 批次大小
    --niter $NITER \                           # 固定学习率的训练轮数
    --niter_decay $NITER_DECAY \               # 学习率线性衰减的训练轮数
    --display_freq 400 \                       # 每400次迭代显示一次结果
    --display_id 1 \                           # Visdom显示窗口ID
    --display_server "http://localhost" \      # Visdom服务器地址
    --display_port 8097 \                      # Visdom服务器端口
    --print_freq 100 \                         # 每100次迭代打印一次损失
    --save_epoch_freq 5 \                      # 每5个epoch保存一次模型
    --sn_gan 1 \                               # 使用谱归一化（SN-CycleGAN项目特色）
    --no_lsgan \                               # 关闭LSGAN，使用Hinge损失
    --wgan 0 \                                 # 关闭WGAN模式
    --with_gp 0 \                              # 关闭梯度惩罚
    --lambda_A 10.0 \                          # A->B->A循环损失权重
    --lambda_B 10.0 \                          # B->A->B循环损失权重
    --lambda_identity 0.05 \                    # 身份映射损失权重
    --csca_training_stage full \               # CSCA训练阶段：full(完整注意力)
    --csca_mid_init_gate 0.05 \                # 中间注意力的初始门控值
    --csca_final_init_gate 0.1 \               # 最终注意力的初始门控值
    --csca_reduction 4 \                       # CSCA通道压缩比例
    --csca_cot_k 3 \                           # CoT注意力的卷积核大小参数
    --csca_l1_weight 1e-4 \                    # CSCA门控的L1正则化权重
    --lambda_csca 0.001                        # CSCA总体损失权重

echo "训练完成！"
echo "模型保存在: ./checkpoints/${DATASET}_csca/"
echo ""
echo "可以使用以下命令测试模型："
echo "python test.py --dataroot $DATAROOT --name ${DATASET}_csca --model cycle_gan --phase test --no_dropout"
