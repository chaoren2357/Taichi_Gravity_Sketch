# Taichi_Gravity_Sketch
以引力做画笔，以宇宙做画布 Using gravity as a brush, and the universe as a canvas
# 团队名
sudo 二仙桥
# 项目名
引力绘图 Gravity Sketch
# 项目介绍
本项目旨在利用Taichi在模拟粒子物理运动方面的计算优势实现一个绘图游戏。
## 灵感来源
看到Taichi中很多项目都是基于粒子系统进行模拟，灵机一动想编写一个游戏让玩家能够自己操控粒子，达成一些目标。联想到游戏《投影寻真》的思路，决定设计玩法为通过吸引粒子形成一个形状，并与关卡目标形状对比打分。
## 实现功能
- 玩家拥有一定数量的粒子可供使用，并有给定的绘图目标（形状轮廓线）。
- 玩家通过鼠标/触摸屏幕操控引力子的位置和引力的打开或关闭来吸引粒子。
- 玩家的游戏目标是将吸引到的粒子所构成形状的与目标轮廓图基本吻合。
## 设计展示
<img width="387" alt="pic1" src="https://user-images.githubusercontent.com/37920501/203885822-c558f7ba-f5d1-4115-ae3f-b901e4ce0ed5.png">
<img width="387" alt="pic2" src="https://user-images.githubusercontent.com/37920501/203885851-92aae210-d01c-4f5c-b525-936eaf61dda3.png">


## 其他
预计前期采用Taichi+Python进行编写，如果后面有时间考虑在前端进行实现
