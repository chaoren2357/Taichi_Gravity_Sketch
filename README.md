# Taichi_Gravity_Sketch  
Gravity Sketch    
引力绘图：以引力做画笔，以宇宙做画布  

# 团队名  
Sudo 二仙桥

# 项目介绍

利用Taichi在模拟粒子物理运动方面的计算优势实现一个绘图游戏。

## 灵感来源

Taichi中很多项目都是基于粒子系统进行模拟，灵机一动想编写一个游戏让玩家能够自己操控粒子，达成一些目标。联想到游戏《投影寻真》的思路，决定设计玩法为通过吸引粒子形成一个形状，并与关卡目标形状对比打分。    
<img width="360" alt="投影寻真游戏图" src="https://user-images.githubusercontent.com/37920501/205470826-14e98d5d-ce0f-40ac-8a93-645f37cc49a9.png">

## 玩法简介

- 场地中有不同种类的粒子以及绘图目标（形状轮廓线）。
- 玩家通过鼠标操控引力子的位置和引力的打开或关闭来吸引粒子。
- 玩家的游戏目标是将吸引到的粒子所构成形状的与相应目标轮廓图基本吻合，达到目标分数（每关不同，100分为完全完成）。
- 鼠标左键构建吸引源，鼠标右键构建排斥源；如果是固定的吸引子，则红色表示排斥，绿色表示吸引，黄色表示不工作。
- 可以自行修改引力大小与阻尼大小。

## 游戏特性

- 算法基于MLS-MPM算法，主要参考[mpm128.py](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm128.py)实现。
- 修改了引力特性，使得距离远的粒子吸引力小。
- 修改了速度规则，增加了阻尼。
- 增加了墙的判断。
- 增加了固定吸引点。
- 增加了不同材料（E，mu）的支持。


## 游戏特性 

# 开始使用
## 安装方式
如果已经有Python3后，可以按下述方法进行安装

```sh
pip3 install -r requirements.txt
python3 main.py
```
## 游戏demo

### 第一关
<img width="360" alt="Level1" src="https://user-images.githubusercontent.com/37920501/205471895-01ec0383-f80b-4c1d-a2a2-6e5835cdb24e.png">

### 第二关
<img width="360" alt="Level1" src="https://user-images.githubusercontent.com/37920501/205471961-dbb2de37-238e-41a2-a542-98ef032297eb.png">

### 第三关
<img width="360" alt="Level2" src="https://user-images.githubusercontent.com/37920501/205471932-b17390c1-4aa5-4925-b262-1552197dea05.png">



