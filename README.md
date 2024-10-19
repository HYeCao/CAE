# Causal Action Empowerment for Efficient Reinforcement Learning in Embodied Agents


This repository is the official PyTorch implementation of **CAE**. 

# 🛠️ Installation Instructions

First, create a virtual environment and install all required packages. 
~~~
conda create -n cae python=3.8
pip install -r requirements.txt
~~~


## 💻 Code Usage

If you would like to run CAE on a standard version of a certain `task`, please use `main_causal.py` to train CAE policies.
~~~
export MUJOCO_GL="osmesa"
~~~
~~~
xvfb-run -a python main_causal.py --env_name task
~~~
If you would like to run CAE on a sparse reward version of a certain `task`, please follow the command below.
~~~
python main_causal.py --env_name task --reward_type sparse
~~~

We also provide the core code of MT-CAE for multi-task learning. MT-CAE is implemented based on the garage package 'https://github.com/rlworkgroup/garage'




## 🙏 Acknowledgement

CAE is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. 
