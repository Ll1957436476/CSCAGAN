# CSCA增强的SN-CycleGAN

本项目将CSCA（通道-空间-上下文注意力）模块集成到SN-CycleGAN（带谱归一化的CycleGAN）的ResNet生成器中，以提高图像到图像转换的性能。

## 项目特色

- **SN + Hinge GAN基础**：使用谱归一化 + Hinge损失的最佳配合
- **CSCA注意力增强**：在生成器中集成先进的注意力机制
- **顶级配置**：采用BigGAN、StyleGAN2等顶会模型验证的最强配置
- **三重优化**：谱归一化 + Hinge损失 + 注意力机制的协同效果

## 主要特性

- **CSCA注意力机制**：在ResNet残差块中集成了双重注意力（中间注意力 + 最终注意力）
- **阶段化训练**：支持渐进式训练策略（mid_only → final_only → full）
- **稳定的门控机制**：保守初始化 + L1正则化，确保训练稳定性
- **完全集成**：无缝集成到原有CycleGAN架构中

## 快速开始

### 1. 测试CSCA集成

```bash
python test_csca_simple.py
```

### 2. 训练CSCA增强的CycleGAN

```bash
# 使用facades数据集
bash scripts/train_cyclegan_csca.sh facades

# 使用horse2zebra数据集
bash scripts/train_cyclegan_csca.sh horse2zebra

# 或者直接使用python命令
python train.py --dataroot ./datasets/facades --name facades_csca --model cycle_gan --netG resnet_9blocks
```

## CSCA参数详细说明

### 核心控制参数
- `--csca_training_stage`: **CSCA训练阶段**
  - `mid_only`: 仅使用中间注意力，适合训练初期
  - `final_only`: 仅使用最终注意力，适合中期训练
  - `full`: 使用完整注意力机制，适合后期训练 (默认)

- `--lambda_csca`: **CSCA总体损失权重** (默认: 0.001)
  - 控制CSCA损失在整体损失中的比重
  - 过大可能影响原始任务，过小则注意力效果不明显

### 注意力强度参数
- `--csca_mid_init_gate`: **中间注意力初始门控值** (默认: 0.05)
  - 控制中间注意力的初始强度
  - 较小值确保训练初期的稳定性

- `--csca_final_init_gate`: **最终注意力初始门控值** (默认: 0.1)
  - 控制最终注意力的初始强度
  - 通常设置比中间注意力稍大

### 架构配置参数
- `--csca_reduction`: **通道压缩比例** (默认: 4)
  - 注意力模块中的通道压缩倍数
  - 较大值减少计算量但可能降低表达能力

- `--csca_cot_k`: **CoT注意力卷积核大小** (默认: 3)
  - 影响上下文注意力的感受野大小
  - 较大值捕获更大范围的上下文信息

- `--csca_l1_weight`: **L1正则化权重** (默认: 1e-4)
  - 门控参数的L1正则化强度
  - 防止注意力权重过度增长

## 训练策略

### 单阶段训练（推荐）
```bash
python train.py --dataroot ./datasets/facades --name facades_csca \
    --csca_training_stage full --lambda_csca 0.001
```

### 阶段化训练（高级）
```bash
# 阶段1：中间注意力 (epoch 0-30)
python train.py --name facades_csca --csca_training_stage mid_only --niter 30

# 阶段2：最终注意力 (epoch 30-60)
python train.py --name facades_csca --csca_training_stage final_only \
    --continue_train --epoch_count 31 --niter 30

# 阶段3：完整注意力 (epoch 60+)
python train.py --name facades_csca --csca_training_stage full \
    --continue_train --epoch_count 61 --niter 40
```

## 架构说明

### CSCA增强的ResNet块结构
```
输入 x
  ↓
第一个卷积块 (Pad → Conv → Norm → ReLU → Dropout?)
  ↓
中间注意力增强 (可选，根据training_stage)
  ↓
第二个卷积块 (Pad → Conv → Norm)
  ↓
最终注意力增强 (可选，根据training_stage)
  ↓
残差连接 (x + 增强特征)
  ↓
输出
```

### 损失函数
总损失 = 原始CycleGAN损失 + λ_csca × CSCA正则化损失

其中CSCA正则化损失包括所有注意力模块的L1门控正则化。

## 文件结构

```
models/
├── CSCA.py                 # CSCA注意力模块实现
├── networks.py             # 修改后的网络定义（集成CSCA）
└── cycle_gan_model.py      # 修改后的CycleGAN模型

scripts/
└── train_cyclegan_csca.sh  # CSCA训练脚本

test_csca_simple.py         # CSCA集成测试脚本
CSCA_README.md             # 本文档
```

## 性能优势

CSCA模块通过以下机制提升CycleGAN性能：

1. **通道注意力**：自适应调节不同通道的重要性
2. **空间注意力**：关注图像中的关键区域
3. **上下文注意力**：捕获长距离依赖关系
4. **双重增强**：中间指导 + 最终增强的协同作用

## 注意事项

1. **内存使用**：CSCA会增加约20-30%的显存使用
2. **训练时间**：相比原始CycleGAN增加约15-25%的训练时间
3. **参数调优**：建议从默认参数开始，根据具体数据集微调

## 故障排除

如果遇到导入错误，请确保：
1. CSCA.py文件在models/目录下
2. 所有依赖包已正确安装
3. Python路径设置正确

如果训练不稳定，可以尝试：
1. 降低lambda_csca权重
2. 使用阶段化训练策略
3. 调整门控初始化值
