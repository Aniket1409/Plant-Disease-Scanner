# Hardware Required
- Any NVIDIA GPU with latest drivers
  
# Prerequisites
- Python 3.9
- Anaconda3 2024.10-1 (64-bit)
- Microsoft Visual C++ 2015-2022 (x64)
- TensorFlow 2.10 (last version with GPU support)

# Steps
1. Clone Repository
2. Installation (Anaconda Prompt) only works if NVIDIA drivers are installed
```bash
nvidia-smi
conda create -n tensorflow_environment python==3.9
conda activate tensorflow_environment
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install --upgrade pip
cd <folder_location>
pip install -r requirements.txt
```

## Note: 
- using pip install > CPU version gets installed
- using requirements.txt > TF detects CUDA installation > installs GPU version
