# Train Object Detector with Detectron2 (Balloon Dataset)

This repository demonstrates training an object detection model on a custom balloon dataset using **Detectron2**.

---

## Table of Contents
- [Environment Setup](#environment-setup)
- [Install Dependencies](#install-dependencies)
- [Download Dataset](#download-dataset)
- [Train the Model](#train-the-model)
- [Evaluate/Use Model](#evaluateuse-model)
- [Notes](#notes)
- [License](#license)

---

## Environment Setup

It is recommended to use a virtual environment to avoid conflicts:

### Using Python `venv`:

```bash
python3 -m venv detectron2-env
source detectron2-env/bin/activate   # On Mac/Linux
detectron2-env\Scripts\activate     # On Windows

pip install --upgrade pip

# Install PyTorch (CPU or GPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install other dependencies
pip install -r requirements.txt

curl -L -o balloon_dataset.zip https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip balloon_dataset.zip

python train.py --device mps --learning-rate 0.00001 --iterations 60

