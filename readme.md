# MRL-RB简介 #


## 运行 ##
- Windows PowerShell 中运行 `./run_training.cmd` 即可看到 5×5 网络的训练过程。


## agent.py ##
- 该文件夹定义了智能体的一些属性和方法，包括神经网络，动作选择函数，训练函数等等。

## buoy.py ##

- 该文件夹定义了浮标、浮标网络、数据包三个类。

## environment.py ##

- 环境的定义，获取观测，执行动作，生成数据包，网络刷新等函数都在里面。

## replay_buffer.py ##

- 定义了两个经验回放池。

## runner.py ##

- 强化学习的执行文件，主要是智能体与环境交互的过程，也包括奖励值回溯机制。

## 安装与运行（建议） ##

- 环境建议使用 Python 3.8 或 3.9。先创建虚拟环境并激活：

- 安装依赖（见 `requirements.txt`）：

	pip install -r requirements.txt

- 运行 `run_training.cmd` 可开始训练；也可以在命令末尾追加参数，例如 `./run_training.cmd --num_env_steps 10000`。

## 项目结构 ##

```text
src/
  buoy_network/       浮标环境、网络、训练入口和执行器
    agents/           DQN、IQL、VDN、QMIX 等备用智能体
  onpolicy/           MAPPO / Graph-MAPPO 算法框架
scripts/              日志分析和绘图脚本
outputs/
  TG-MAPPO/          TG-MAPPO 运行产生的新日志和训练曲线
    diagnostic_logs/ 网络诊断日志
    training_logs/   训练指标日志
    plots/           按时间生成的训练曲线
  diagnostic_logs/    浮标网络诊断日志
  training_logs/      训练日志
  plots/              每次训练生成的曲线
  figures/            对比图和历史图片
run_training.cmd      Windows 一键训练入口
requirements.txt      Python 依赖
```

## 依赖（候选） ##

项目使用到的主要第三方库（请根据平台与显卡调整版本）：

- torch
- numpy
- matplotlib
- scipy
- gym
- torch-geometric
- wandb

注：某些库（如 `torch-geometric`）安装较复杂，请参考各自官方安装说明。

