# Project Status Report

**Date:** 2025年8月9日

## 1. Overall Summary

This project is an advanced implementation of CycleGAN designed for high-quality style transfer. The core of the project is SN-CycleGAN (CycleGAN with Spectral Normalization), which has been significantly enhanced by a custom-designed attention module named **CSCA (Channel-Spatial-Context Attention)**.

This document outlines the final architecture (**v4.1**) after a series of intensive, collaborative design and debugging sessions. The model now incorporates numerous state-of-the-art mechanisms and critical reliability fixes to maximize performance, stability, and expressiveness.

## 2. Core Architecture: CSCA v4.1

The final attention module, integrated into the generator's ResNet blocks, consists of two main stages:

-   **Mid-Attention Stage**: A lightweight attention block featuring **ECA (Efficient Channel Attention)** to provide an efficient, early-stage feature calibration.
-   **Final-Attention Stage**: A powerful and highly configurable main attention block that fuses two sophisticated sub-modules:
    1.  **Coord2H Attention**: A novel **Dual-Head Coordinate Attention** mechanism, now featuring selectable fusion paths and a stabilizing frontend.
    2.  **Advanced CoT Attention**: A highly modified **Contextual Transformer** attention block, now with enhanced numerical stability.

## 3. Key Innovations and Reliability Fixes

The final v4.1 model is the culmination of the following key improvements:

1.  **Upgraded Coord2H Dual-Head Attention**: The coordinate attention module was significantly enhanced with:
    *   **Decoupled Heads**: Separate convolutional heads for horizontal and vertical patterns (`conv_h`, `conv_w`).
    *   **Selectable Amplification**: A choice between a stable `sigmoid` fusion path and a linear path (`w = 1+κ(w_lin-1)`) that can amplify features, controlled by `--coord_use_softmax` and `--coord_kappa`.
    *   **Frontend Normalization**: A projection head (`1x1 Conv + IN + SiLU`) was added to stabilize the inputs to the dual heads.

2.  **Robust CoT Attention Dynamics**: The CoT branch was made more powerful and reliable:
    *   **Advanced Gating**: Uses **(a) Temperature Scaling (τ)**, **(b) Mean Rescaling**, and **(c) Hybrid Gating (λ)** for fine-grained control over attention sharpness and strength.
    *   **Numerical Stability**: The softmax calculation is now performed in `fp32` and stabilized by subtracting the max value, preventing overflow in mixed-precision training.
    *   **Corrected Broadcasting**: A critical bug was fixed to ensure correct tensor shapes during attention application.
    *   **Hyperparameter Safety**: `τ` and `λ` values are now clamped to safe ranges during the forward pass.

3.  **Implicit Multi-Head Design**: Both the `Coord2H` and `CoT` branches use **grouped convolutions** (`groups=h`) to simulate multi-head attention, promoting feature disentanglement.

4.  **Corrected Residual Scaling**: A previously-missed bug was fixed. Each `ResnetBlock` now correctly applies a learnable scalar (`res_scale`) to its output, a proven technique to enhance training stability.

5.  **Full Affine Normalization & Other Best Practices**: Includes `affine=True` on all `InstanceNorm` layers, use of `register_buffer` for non-parameter state, safe channel reduction, and type-safe loss calculations.

## 4. Current Status & Next Steps

**Current Status**: The model architecture is now finalized at v4.1. The codebase is feature-complete, robust, and highly configurable via an extensive set of command-line arguments.

**CRITICAL: Checkpoint Incompatibility**
> The numerous architectural changes mean that **all previously saved model checkpoints are now incompatible with the updated code.** A full retraining is mandatory.

**Next Steps**:
1.  **Train & Evaluate**: The immediate and most critical next step is to launch a new training run from scratch to evaluate the performance of the final v4.1 model.
2.  **Ablation Studies**: To provide strong evidence for a research paper, design ablation studies that systematically enable/disable the key innovations to quantify their impact.
3.  **Hyperparameter Tuning**: Experiment with the new command-line arguments to find the optimal configuration for different datasets.

## 5. How to Train

To start a new training run, use the existing scripts or run `train.py` directly. Remember to use a new experiment name and the new available flags for experimentation.

```bash
# Example for a sharp, directional style like horse2zebra, using the new amplification path
python train.py --dataroot ./datasets/horse2zebra \
                --name horse2zebra_csca_v4.1 \
                --model cycle_gan \
                --netG resnet_9blocks \
                --csca_cot_heads 8 \
                --coord_use_softmax \
                --coord_tau 1.0 \
                --coord_kappa 1.0 \
                --csca_cot_temp 0.8 \
                --csca_cot_lambda 0.7
```
```
