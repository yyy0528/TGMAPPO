@echo off
setlocal

set "PYTHON_EXE=D:\python3.6.8\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python interpreter not found: %PYTHON_EXE%
    echo Edit PYTHON_EXE in run_training.cmd to point to the correct interpreter.
    exit /b 1
)

cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

"%PYTHON_EXE%" run_algorithm.py --algorithm tg-mappo ^
  --project_name TG-MAPPO ^
  --env_name GraphMPE ^
  --algorithm_name mappo ^
  --seed 0 ^
  --experiment_name TG-MAPPO ^
  --scenario_name navigation_graph ^
  --num_agents 25 ^
  --collision_rew 5 ^
  --n_training_threads 1 ^
  --n_rollout_threads 1 ^
  --num_mini_batch 20 ^
  --episode_length 80 ^
  --num_env_steps 200 ^
  --ppo_epoch 10 ^
  --use_ReLU ^
  --gain 0.01 ^
  --lr 5e-4 ^
  --critic_lr 3e-4 ^
  --user_name marl ^
  --use_cent_obs False ^
  --graph_feat_type global ^
  --target_mini_batch_size 20 ^
  %*

exit /b %ERRORLEVEL%
