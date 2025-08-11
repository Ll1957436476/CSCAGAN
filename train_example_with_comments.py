#!/usr/bin/env python3
"""
CSCA增强的CycleGAN训练示例
带有详细的中文参数注释
"""

import os
import sys

def train_csca_cyclegan():
    """训练CSCA增强的CycleGAN模型"""
    
    # 基础训练参数
    base_cmd = [
        "python", "train.py",
        
        # === 数据集相关参数 ===
        "--dataroot", "./datasets/facades",        # 数据集路径
        "--name", "facades_csca",                  # 实验名称，用于保存模型
        
        # === 模型架构参数 ===
        "--model", "cycle_gan",                    # 使用CycleGAN模型
        "--netG", "resnet_9blocks",                # 生成器：9层ResNet块架构
        "--netD", "basic",                         # 判别器：基础PatchGAN
        
        # === 训练配置参数 ===
        "--pool_size", "50",                       # 图像池大小，存储历史生成图像
        "--no_dropout",                            # 禁用dropout（CycleGAN默认）
        "--gpu_ids", "0",                          # 使用的GPU设备ID
        "--batch_size", "1",                       # 批次大小
        
        # === 训练轮数参数 ===
        "--niter", "100",                          # 固定学习率的训练轮数
        "--niter_decay", "100",                    # 学习率线性衰减的训练轮数
        
        # === 显示和保存参数 ===
        "--display_freq", "400",                   # 每400次迭代显示结果
        "--print_freq", "100",                     # 每100次迭代打印损失
        "--save_epoch_freq", "5",                  # 每5个epoch保存模型

        # === SN-CycleGAN核心参数 ===
        "--sn_gan", "1",                           # 使用谱归一化（项目特色）
        "--gan_mode", "hinge",                     # 使用Hinge损失（与SN的最佳配合）
        "--lambda_A", "10.0",                      # A->B->A循环损失权重
        "--lambda_B", "10.0",                      # B->A->B循环损失权重
        "--lambda_identity", "0.5",                # 身份映射损失权重

        # === CSCA核心参数 ===
        "--csca_training_stage", "full",           # CSCA训练阶段：完整注意力
        "--lambda_csca", "0.001",                  # CSCA损失权重
        
        # === CSCA注意力强度参数 ===
        "--csca_mid_init_gate", "0.05",           # 中间注意力初始门控值
        "--csca_final_init_gate", "0.1",          # 最终注意力初始门控值
        
        # === CSCA架构参数 ===
        "--csca_reduction", "4",                   # 通道压缩比例
        "--csca_cot_k", "3",                       # CoT注意力卷积核大小
        "--csca_l1_weight", "1e-4",               # L1正则化权重
    ]
    
    # 执行训练命令
    cmd_str = " ".join(base_cmd)
    print("=== 开始训练CSCA增强的CycleGAN ===")
    print(f"执行命令: {cmd_str}")
    print()
    
    os.system(cmd_str)

def train_staged_csca():
    """阶段化训练CSCA模型"""
    
    print("=== 阶段化训练策略 ===")
    
    # 阶段1：仅中间注意力 (0-30 epochs)
    print("阶段1: 中间注意力训练 (0-30 epochs)")
    stage1_cmd = [
        "python", "train.py",
        "--dataroot", "./datasets/facades",
        "--name", "facades_csca_staged",
        "--model", "cycle_gan",
        "--netG", "resnet_9blocks",
        "--no_dropout",
        "--niter", "30",                           # 30个epoch
        "--niter_decay", "0",                      # 不衰减学习率
        "--sn_gan", "1",                           # 使用谱归一化
        "--gan_mode", "hinge",                     # 使用Hinge损失
        "--lambda_A", "10.0",                      # A->B->A循环损失权重
        "--lambda_B", "10.0",                      # B->A->B循环损失权重
        "--lambda_identity", "0.5",                # 身份映射损失权重
        "--csca_training_stage", "mid_only",       # 仅中间注意力
        "--lambda_csca", "0.001",
        "--csca_mid_init_gate", "0.05",
        "--csca_final_init_gate", "0.1",
        "--csca_reduction", "4",
        "--csca_cot_k", "3",
        "--csca_l1_weight", "1e-4",
    ]
    os.system(" ".join(stage1_cmd))
    
    # 阶段2：仅最终注意力 (30-60 epochs)
    print("阶段2: 最终注意力训练 (30-60 epochs)")
    stage2_cmd = [
        "python", "train.py",
        "--dataroot", "./datasets/facades",
        "--name", "facades_csca_staged",
        "--model", "cycle_gan",
        "--netG", "resnet_9blocks",
        "--no_dropout",
        "--continue_train",                        # 继续训练
        "--epoch_count", "31",                     # 从第31个epoch开始
        "--niter", "30",                           # 再训练30个epoch
        "--niter_decay", "0",
        "--sn_gan", "1",                           # 使用谱归一化
        "--gan_mode", "hinge",                     # 使用Hinge损失
        "--lambda_A", "10.0",                      # A->B->A循环损失权重
        "--lambda_B", "10.0",                      # B->A->B循环损失权重
        "--lambda_identity", "0.5",                # 身份映射损失权重
        "--csca_training_stage", "final_only",     # 仅最终注意力
        "--lambda_csca", "0.001",
        "--csca_mid_init_gate", "0.05",
        "--csca_final_init_gate", "0.1",
        "--csca_reduction", "4",
        "--csca_cot_k", "3",
        "--csca_l1_weight", "1e-4",
    ]
    os.system(" ".join(stage2_cmd))
    
    # 阶段3：完整注意力 (60+ epochs)
    print("阶段3: 完整注意力训练 (60+ epochs)")
    stage3_cmd = [
        "python", "train.py",
        "--dataroot", "./datasets/facades",
        "--name", "facades_csca_staged",
        "--model", "cycle_gan",
        "--netG", "resnet_9blocks",
        "--no_dropout",
        "--continue_train",                        # 继续训练
        "--epoch_count", "61",                     # 从第61个epoch开始
        "--niter", "40",                           # 固定学习率40个epoch
        "--niter_decay", "60",                     # 衰减学习率60个epoch
        "--sn_gan", "1",                           # 使用谱归一化
        "--gan_mode", "hinge",                     # 使用Hinge损失
        "--lambda_A", "10.0",                      # A->B->A循环损失权重
        "--lambda_B", "10.0",                      # B->A->B循环损失权重
        "--lambda_identity", "0.5",                # 身份映射损失权重
        "--csca_training_stage", "full",           # 完整注意力
        "--lambda_csca", "0.001",
        "--csca_mid_init_gate", "0.05",
        "--csca_final_init_gate", "0.1",
        "--csca_reduction", "4",
        "--csca_cot_k", "3",
        "--csca_l1_weight", "1e-4",
    ]
    os.system(" ".join(stage3_cmd))

def main():
    """主函数"""
    print("CSCA增强的CycleGAN训练示例")
    print("1. 单阶段训练（推荐）")
    print("2. 阶段化训练（高级）")
    
    choice = input("请选择训练方式 (1/2): ").strip()
    
    if choice == "1":
        train_csca_cyclegan()
    elif choice == "2":
        train_staged_csca()
    else:
        print("无效选择，使用默认单阶段训练")
        train_csca_cyclegan()

if __name__ == "__main__":
    main()
