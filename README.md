# WDMS-Net

Official PyTorch implementation of **"WDMS-Net: A Wavelet Downsampling and Multi-Scale Convolution Network for Step-Wise mmWave Gait Recognition"**, published in *[Journal of Infrared, Millimeter, and Terahertz Waves](https://link.springer.com/article/10.1007/s10762-026-01174-9)* ([free view-only version](https://rdcu.be/3nSobhGR9XWa)).

WDMS-Net performs gait recognition on mmWave radar time–Doppler images by replacing conventional pooling with wavelet downsampling (WD) and adopting a multi-scale (MS) convolution block, achieving 98.00% average accuracy (10-fold cross-validation) with 21.7 MB of parameters.

## Repository Structure

    WDMS-Net
    ├── Main/                          # Main code for model training and validation in the paper
    │   ├── train_valid.py             # Training & validation
    │   ├── validation_result.py       # Per-class accuracy evaluation
    │   └── checkout_saveload.py       # Checkpoint save/load utilities
    ├── Model/                         # Models used in the main-module ablation study (Table 5)
    │   ├── VGG19.py                   # VGG19 baseline
    │   ├── VGG19_FC.py                # VGG19 with modified FC layers
    │   ├── VGG19_FC_WD.py             # VGG19_FC + wavelet downsampling
    │   ├── WDMS_Net.py                # Proposed WDMS-Net
    │   ├── WaveDown_Block.py          # Wavelet downsampling module (WD)
    │   └── MultiScale_Layer.py        # Multi-scale convolution module (MS)
    └── Experiment/                    # Supplementary experiments
        ├── Finetuning.py              # Cross-environment fine-tuning
        └── 10fold_cross_validation/
            ├── cross_validation.py    # Stratified 10-fold cross-validation
            └── ten_fold_log/          # TensorBoard logs & results (seeds: 49 / 88 / 520 / 2026)

## Requirements

- Python 3.8, PyTorch (CUDA recommended)
- Additional dependencies: `torchvision`, `pytorch_wavelets`, `thop`, `tensorboard`, `scikit-learn`, `Pillow`, `NumPy`

    pip install torch torchvision pytorch_wavelets thop tensorboard scikit-learn pillow numpy

## Usage

Each script is self-contained; dataset and checkpoint paths are hard-coded and should be modified before running.

Data should be organized in the standard `ImageFolder` layout (folder names as class labels), with images resized to 299×299 and normalized using the training-set statistics.

    # Training & validation (paper model)
    python Main/train_valid.py

    # Per-class accuracy evaluation
    python Main/validation_result.py

    # Stratified 10-fold cross-validation
    python Experiment/10fold_cross_validation/cross_validation.py

    # Cross-environment fine-tuning
    python Experiment/Finetuning.py

    # Visualize the 10-fold cross-validation training curves
    tensorboard --logdir Experiment/10fold_cross_validation/ten_fold_log

The training logs of the 10-fold cross-validation (TensorBoard event files) are provided in `Experiment/10fold_cross_validation/ten_fold_log/`. After running the `tensorboard` command above, open http://localhost:6006 in your browser to view the training curves.

Random seeds are fixed (`seed=49` by default) for reproducibility; the 10-fold results under four seeds are provided in `ten_fold_log/`.

## Citation

If this work is helpful to your research, it would be our greatest honor. We would greatly appreciate it if you could consider citing our paper:

    @article{wu2026wdmsnet,
      title={WDMS-Net: A Wavelet Downsampling and Multi-Scale Convolution Network for Step-Wise mmWave Gait Recognition},
      author={Wu, Ruipeng and Sun, Zhiyuan and Sun, Zeyue and Tu, Hao and Wang, Tao},
      journal={Journal of Infrared, Millimeter, and Terahertz Waves},
      year={2026},
      doi={10.1007/s10762-026-01174-9}
    }

## Contact

If you have any questions regarding the code, or need additional code that has not yet been released, please feel free to contact us at ruipeng_wu@mail.hfut.edu.cn. We are always happy to help.

The gait dataset used in this study is still involved in our ongoing research, and therefore cannot be made fully public at this stage. It is available upon reasonable request: please kindly obtain permission from the corresponding author first, and then feel free to reach us at ruipeng_wu@mail.hfut.edu.cn to obtain the data. We sincerely appreciate your understanding.

## License

This project is released under the [MIT License](LICENSE).