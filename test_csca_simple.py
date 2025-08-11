#!/usr/bin/env python3
"""
简单的CSCA集成测试脚本
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append('.')

def test_basic_import():
    """测试基本导入"""
    print("=== 测试基本导入 ===")
    try:
        from models.CSCA import CSCAConfig
        print("✓ CSCA模块导入成功")
        
        from models.networks import ResnetBlock
        print("✓ ResnetBlock导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_resnet_block():
    """测试CSCA增强的ResNet块"""
    print("\n=== 测试CSCA增强的ResNet块 ===")
    try:
        from models.networks import ResnetBlock
        from models.CSCA import CSCAConfig
        import torch.nn as nn
        
        # 创建CSCA配置
        csca_config = CSCAConfig()
        
        # 创建CSCA增强的ResNet块
        block = ResnetBlock(
            dim=256,
            padding_type='reflect',
            norm_layer=nn.InstanceNorm2d,
            use_dropout=False,
            use_bias=True,
            csca_config=csca_config,
            training_stage='full'
        )
        print("✓ CSCA增强ResNet块创建成功")
        
        # 测试前向传播
        x = torch.randn(1, 256, 64, 64)
        with torch.no_grad():
            out = block(x)
        
        print(f"✓ 前向传播成功，输入: {x.shape}, 输出: {out.shape}")
        
        # 测试训练阶段切换
        block.set_training_stage('mid_only')
        print("✓ 训练阶段切换成功")
        
        # 测试损失获取
        _ = block(x)  # 前向传播生成损失
        loss = block.get_csca_losses()
        print(f"✓ CSCA损失获取成功: {loss}")
        
        return True
    except Exception as e:
        print(f"✗ ResNet块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generator():
    """测试CSCA增强的生成器"""
    print("\n=== 测试CSCA增强的生成器 ===")
    try:
        from models.networks import define_G
        from models.CSCA import CSCAConfig
        
        # 创建CSCA配置
        csca_config = CSCAConfig()
        
        # 创建CSCA增强的生成器
        netG = define_G(
            input_nc=3,
            output_nc=3,
            ngf=64,
            netG='resnet_6blocks',  # 使用较小的网络进行测试
            norm='instance',
            use_dropout=False,
            init_type='normal',
            init_gain=0.02,
            gpu_ids=[],
            csca_config=csca_config,
            training_stage='full'
        )
        print("✓ CSCA增强生成器创建成功")
        
        # 测试前向传播
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = netG(x)
        
        print(f"✓ 生成器前向传播成功，输入: {x.shape}, 输出: {out.shape}")
        
        # 测试训练阶段设置
        netG.set_csca_training_stage('mid_only')
        print("✓ 生成器训练阶段设置成功")
        
        # 测试损失获取
        _ = netG(x)  # 前向传播生成损失
        total_loss = netG.get_csca_losses()
        print(f"✓ 生成器CSCA损失获取成功: {total_loss}")
        
        return True
    except Exception as e:
        print(f"✗ 生成器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始CSCA集成简单测试...\n")
    
    tests = [
        test_basic_import,
        test_resnet_block,
        test_generator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"测试异常: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！CSCA集成成功！")
        print("\n现在你可以使用以下命令训练CSCA增强的CycleGAN：")
        print("python train.py --dataroot ./datasets/facades --name facades_csca --model cycle_gan --netG resnet_9blocks")
        return True
    else:
        print("❌ 部分测试失败，请检查错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
