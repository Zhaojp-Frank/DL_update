# A generic communicaiton scheduler for distributed DNN training
Yanghua Peng etc, SOSP19

# 1.Background
communication scheduling是新涌现的优化多机训练加速的方法，例如[TicTac SysML19] 和 [P3, SysML19]. 不影响结果准确性的同时，取得了不错的加速效果。
- 思路：优先传输那些DNN前面的layer的梯度 使得下一轮的计算可以尽快开始；而后续层的传输可以与计算并行。减少计算等待

但有一些不足：
## 不够通用，修改了很多现有的框架
对此提出通用的通信调度优化，考虑到一些框架如(TF, PyTorch)采用了global barrier机制(batch之间显式同步)，因此设计了两种: dependency proxy 和 layer-wise out-of-engine两种

## 对不同系统配置适应性不足
例如PS架构，参数最好拆分，然后采用push+pull来充分利用双向带宽；而AllRedcue已经拆分到layer，最好能聚合来提高传送效率。
为此 提出profile-analze 就贝叶斯优化来自动寻找最佳参数

重点是：**通用的**通信优化（适用TF PyTorch MxNET, PS or AllReduce, TCP/RDMA, 多种模型)， 效果超越SOTA。例如比P3快28%-43%

# 2. 简介
