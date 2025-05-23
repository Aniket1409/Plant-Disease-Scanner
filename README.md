# Prerequisites
- Python 3.9
- Anaconda3 2024.10-1 (64-bit)
- Microsoft Visual C++ 2015-2022 (x64)
- TensorFlow 2.10 (last version with GPU support)
- NVIDIA GPU with latest drivers

# Steps
1. Clone Repository

2. Create Tensorflow Environment (only works if NVIDIA drivers are installed)
```cmd
nvidia-smi
conda create -n tensorflow_environment python==3.9
conda activate tensorflow_environment
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install --upgrade pip
cd <folder_location>
pip install -r requirements.txt
```

- using pip install → CPU version gets installed
- using requirements.txt → TF detects CUDA installation → installs GPU version

3. Run Using Streamlit
- Open cloned repository in code editor → open app.py
- Select Interpreter → tensorflow_environment
```cmd
streamlit run app.py
```
