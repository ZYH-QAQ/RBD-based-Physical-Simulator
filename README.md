# RBD-based-Physical-Simulator
A physical-simulation programme based on rigid body dynamics, implemented to solve a specific problem in which a rigid body is hanged on a spot using 3 springs attached to it. The problem came from 华中科技大学数学建模竞赛国赛选拔 problem A.

### 使用说明：
balance.cpp求解刚体平衡位置。solution.cpp模拟刚体运动。

本程序通过leapfrog的方法实现欧拉半隐式积分，使结果拥有dt的二阶精确性，部分向量和矩阵运算用openmp加速。

与一般物理引擎采用的描述旋转的四元数不同，我采用了旋转矩阵R来描述刚体的旋转，质心的位矢来描述刚体的平动坐标，辅助的物理量包括质心速度矢量、地面坐标架的角速度矢量和力、力矩（相对质心）矢量等。
### 具体在每一帧的更新中：

1.根据[0]时刻的刚体位形，求出刚体受力和力矩。

2.将惯量主轴坐标架中刚体的惯量张量，用旋转矩阵变换到地面坐标架上。根据角动量定理求出角加速度，其中旋转矩阵和旋转矩阵的导数取作R[0]和dR/dt|[0]。根据牛顿第二定律求出质心加速度。

3.根据角加速度和质心加速度更新角速度和质心速度。（在求解平衡位置的模式下，额外增加一个速度衰减项）

4.根据角速度更新旋转矩阵，根据质心速度更新质心位矢。
