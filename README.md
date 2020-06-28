# Applying TD3 Algorithm for Shake Table System Control

Shake table simulation is regarded as one of the most straightforward experimental approach to evaluate the seismic resistance of the structure. There is a dynamical coupling between the structure and the shake table. Unmodeled or mismodeled interaction would cause poor control performance. Our goal is to apply deep reinforcement learning (DRL) to interact with custom shaking table environment, so that we could solve the situation above through the training result.

## Getting Started

```
git clone https://github.com/SJ-Chuang/TD3-Shake-Table-System.git
cd TD3-Shake-Table-System
```

## Train

```
python3 td3.py --env_dir ./env
```

```
Optional arguments
    --env_dir,		environment directory,			default='./env'
    --test,			test mode
    --do_render,	do render
```

## Hyper-parameter tuning

Modify the hyper-parameters in \__main__ of td3.py

changeable parameter:

​	**batch size**,

​	**exploration noise**,

​	**learning rate**,

​	**discount factor**,

​	**interpolation factor**

## Save plot to current directory

```python
env.plot_history(name = 'filename.jpg')
```

![image](figure/figure01.jpg)