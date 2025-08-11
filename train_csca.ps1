# CSCA增强的CycleGAN训练脚本 (PowerShell版本)
# 使用方法：.\train_csca.ps1 [数据集名称]

param(
    [string]$Dataset = "horse2zebra"
)

$DATAROOT = "./datasets/$Dataset"
$GPU_IDS = "0"
$BATCH_SIZE = 1
$NITER = 25
$NITER_DECAY = 25

Write-Host "=== 训练CSCA增强的CycleGAN模型 ===" -ForegroundColor Green
Write-Host "数据集: $Dataset" -ForegroundColor Yellow
Write-Host "数据路径: $DATAROOT" -ForegroundColor Yellow

# 检查数据集是否存在
if (-not (Test-Path $DATAROOT)) {
    Write-Host "错误: 数据集路径不存在: $DATAROOT" -ForegroundColor Red
    Write-Host "请确保数据集已正确放置在datasets文件夹中" -ForegroundColor Red
    exit 1
}

# 执行训练命令
python train.py `
    --dataroot $DATAROOT `
    --name "${Dataset}_csca" `
    --model cycle_gan `
    --netG resnet_9blocks `
    --pool_size 50 `
    --no_dropout `
    --gpu_ids $GPU_IDS `
    --batch_size $BATCH_SIZE `
    --niter $NITER `
    --niter_decay $NITER_DECAY `
    --display_freq 400 `
    --display_id 1 `
    --display_server "http://localhost" `
    --display_port 8097 `
    --print_freq 100 `
    --save_epoch_freq 5 `
    --sn_gan 1 `
    --no_lsgan `
    --wgan 0 `
    --with_gp 0 `
    --lambda_A 10.0 `
    --lambda_B 10.0 `
    --lambda_identity 0.5 `
    --csca_training_stage full `
    --csca_mid_init_gate 0.05 `
    --csca_final_init_gate 0.1 `
    --csca_reduction 4 `
    --csca_cot_k 3 `
    --csca_l1_weight 1e-4 `
    --lambda_csca 0.001

if ($LASTEXITCODE -eq 0) {
    Write-Host "训练完成！" -ForegroundColor Green
    Write-Host "模型保存在: ./checkpoints/${Dataset}_csca/" -ForegroundColor Green
} else {
    Write-Host "训练过程中出现错误，退出码: $LASTEXITCODE" -ForegroundColor Red
}

Write-Host "按任意键继续..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
