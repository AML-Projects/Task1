# Task1

# INSTALLATION GUIDE FOR TASK 1

Windows 10 - 64-Bit

Requirments: **Patience & nerves**
Virtual environment disk space: 

## 1 Install/Update Anaconda

- We use Python 3.8, if older version installed:
- Run Anaconda Prompt (Located in Start/Anaconda3/Anaconda Prompt) as Admin (Right click, run as Administrator)
- `conda install -c anaconda python=3.8`
- `python --version` should be Python 3.8.5 now

### 1.1 Create conda environement
- `conda create --prefix Path/to/env python=3.8
- Path/to/env could be project dir. the name of the env will be the last foldername (here env)
- Activate env: `conda activate path/to/env`
- Check if python is pointing to the new environment: `where python` (macOS would be `which python`)
	- Path to new environment should be listed
- Check if environment active: `conda info --envs`

### 2.2 Install requirements
- Make sure your env is activated
- `conda list`shows currently installed packages
- Run `conda install pandas=1.1.2`
- Run `conda install scikit-learn=0.23.2`

## 3 Add new Project in PyCharm
- Open Pycharm: File > New Project
- Set Location to GitRepo
- Project Interpreter: Existing Interpreter, Browse for your virtualenv i.e. `D:\GitHub\AML\Task1\aml\python.exe`
- Create
