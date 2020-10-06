# Task1

# INSTALLATION GUIDE FOR TASK 1 - Windows

Windows 10 - 64-Bit

Requirments: **Patience & nerves**
Virtual environment disk space: ~900MB

- Use windows cmd to complete the follwing tasks

## 1 Install/Update Anaconda
- We use Python 3.8, if older version installed:
- `conda install -c anaconda python=3.8`
- `python --version` should be Python 3.8.5 now

### 1.1 Create conda environement
- `conda create -n myenv python=3.8
- Activate env: `conda activate myenv`
- Check if python is pointing to the new environment: `where python` (macOS would be `which python`)
	- Path to new environment should be listed
- Check if environment active: `conda info --envs`

## 2 Install Packages

### 2.2 Install requirements
- Make sure your env is activated
- `conda list`shows currently installed packages
- Run `conda install pandas=1.1.2`
- Run `conda install scikit-learn=0.23.2`

## 3 Add new Project in PyCharm
- Open Pycharm: File > New Project
- Set Location to GitRepo
- Project Interpreter: Existing Interpreter, Browse for your conda env i.e. `D:\GitHub\AML\Task1\aml\python.exe`
- Create
