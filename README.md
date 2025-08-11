# SNCycleGAN: CycleGAN with Spectral Normalization and CSCA

This project is a PyTorch implementation of CycleGAN for unpaired image-to-image translation, with integrated Spectral Normalization (SN) and a Cross-Scale Correspondence Attention (CSCA) module. It builds upon the official [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) implementation.

## Key Features

-   **Spectral Normalization (SN)**: The discriminators are enhanced with Spectral Normalization for more stable training dynamics.
-   **Cross-Scale Correspondence Attention (CSCA)**: A novel attention module designed to build better correspondence between input and output images, improving translation quality.
-   **Flexible Framework**: Easily train and test on new datasets.
-   **Visdom Integration**: Monitor training progress, including images and loss plots, through Visdom.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ll1957436476/CSCAGAN.git
    cd CSCAGAN
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment. All dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Download a Dataset

Use the provided script to download one of the standard CycleGAN datasets (e.g., `horse2zebra`).

```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

### 2. Training

Training scripts are highly customizable through command-line arguments. Here is an example of how to train a model on the `horse2zebra` dataset.

```bash
# Start a visdom server for visualization (in a separate terminal)
# python -m visdom.server

# Train the model
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_csca_model --model cycle_gan --display_id 1
```

-   `--dataroot`: Path to your dataset.
-   `--name`: Name for the experiment. Checkpoints and logs will be saved to `./checkpoints/[name]`.
-   `--model`: Specifies which model to use (`cycle_gan` in this case).
-   `--display_id`: Visdom window ID. Set to `0` or a negative number to disable visdom.

### 3. Testing

Once the model is trained, you can test it on a set of images.

```bash
python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_csca_model --model cycle_gan --phase test
```

The test results will be saved in the `./results/[name]` directory.

## License

This project is licensed under the BSD License. See the `LICENSE` file for more details.

## Acknowledgments

-   This code is heavily based on the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.
-   The core idea is based on the paper [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) by Zhu et al.