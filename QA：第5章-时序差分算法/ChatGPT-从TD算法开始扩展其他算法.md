# 从TD算法开始扩展其他算法

**User:** yeli yin (psmanyyl@gmail.com)  
**Exported:** 2026/1/23 20:33:21  
**Link:** [https://chatgpt.com/c/697367ba-f51c-832d-ae8f-2f584be1be68](https://chatgpt.com/c/697367ba-f51c-832d-ae8f-2f584be1be68)  

## Prompt:
# 请你扮演一个强化学习领域老师的角色，详细的讲解强化学习领域的知识和工程问题 
以理论+伪代码的方式为我解释【TD(λ)算法】，然后给出对应python代码实现，代码要有非常详细的代码注释，辅助初学者理解代码构造和执行逻辑

## Response:
下面我将以**强化学习课程教师**的视角，系统、循序渐进地讲解 **TD(λ)（Temporal Difference with Eligibility Traces）算法**，内容包括：

1.  TD 学习的背景与动机
2.  TD(0) 的局限性
3.  Eligibility Trace（资格迹）的直觉与数学定义
4.  **TD(λ) 的完整理论推导**
5.  **TD(λ) 的伪代码（教学版）**
6.  **TD(λ) 的 Python 实现（含非常详细的逐行注释）**
7.  工程实现中的常见问题与经验建议

* * *

一、背景：强化学习中的价值函数学习
-----------------

在强化学习中，我们通常希望学习**状态价值函数**：

$$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s \right]
$$

当 **环境模型未知** 时，我们无法使用动态规划（DP），只能通过采样轨迹进行学习，这就引出了：

*   **Monte Carlo（MC）方法**
*   **Temporal Difference（TD）方法**

* * *

二、TD(0)：一步时序差分学习
----------------

### TD(0) 更新公式

$$
V(s_t) \leftarrow V(s_t) + \alpha \delta_t
$$

其中 TD 误差定义为：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### TD(0) 的优点

*   不需要等到 episode 结束
*   在线、低方差

### TD(0) 的问题

*   **只更新当前状态**
*   信号传播慢（reward 需要一步一步往前“爬”）

* * *

三、核心思想：Eligibility Trace（资格迹）
-----------------------------

### 直觉解释（非常重要）

> **最近访问过的状态，应该对当前 TD 误差负责，但责任随时间衰减**

也就是说：

*   当前 TD 误差不只更新  $s_t$ 
*   还要更新  $s_{t-1}, s_{t-2}, \dots$ 
*   但越早访问的状态，更新幅度越小

* * *

### 资格迹的数学定义

对每一个状态  $s$ ，维护一个资格迹：

$$
e_t(s)
$$

更新规则为：

$$
e_t(s) = \begin{cases} \gamma \lambda e_{t-1}(s) + 1, & \text{如果 } s = s_t \\ \gamma \lambda e_{t-1}(s), & \text{否则} \end{cases}
$$

其中：

*    $\gamma$ ：折扣因子
*    $\lambda \in [0,1]$ ：**trace decay 参数**

* * *

四、TD(λ) 的完整理论形式
---------------

### TD 误差（仍然是一步 TD）

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### 价值函数更新（对所有状态）

$$
V(s) \leftarrow V(s) + \alpha \delta_t e_t(s)
$$

* * *

### λ 的含义（非常关键）

| λ 值 | 行为特性 |
| --- | --- |
| λ = 0 | 退化为 TD(0) |
| λ → 1 | 接近 Monte Carlo |
| 中间值 | TD 与 MC 的折中 |

**TD(λ) 本质上是一族算法**

* * *

五、TD(λ) 教学伪代码
-------------

```text
初始化 V(s) 为任意值
对每个 episode：
    初始化 eligibility trace e(s) = 0

    初始化状态 s

    while episode 未结束：
        执行动作，得到 r, s'
        
        δ = r + γ V(s') - V(s)

        e(s) = e(s) + 1   # 累加当前状态的资格迹

        对所有状态 s_i：
            V(s_i) ← V(s_i) + α * δ * e(s_i)
            e(s_i) ← γ * λ * e(s_i)

        s ← s'
```

* * *

六、Python 实现（表格状态空间，教学级别）
------------------------

下面的实现假设：

*   **离散状态空间**
*   **已给定策略（Policy Evaluation）**
*   目标是学习 `V(s)`

* * *

### 示例环境假设

*   环境提供：`reset()`、`step(action)`
*   策略是随机策略（仅用于教学）

* * *

### 完整 Python 代码（含极其详细注释）

```python
import numpy as np
import random

class TDLambdaAgent:
    def __init__(self, n_states, alpha=0.1, gamma=0.99, lam=0.8):
        """
        TD(λ) Agent 构造函数

        参数说明：
        n_states : 状态空间大小（离散）
        alpha    : 学习率
        gamma    : 折扣因子
        lam      : λ，资格迹衰减参数
        """

        # 状态价值函数 V(s)，初始化为 0
        self.V = np.zeros(n_states)

        # eligibility trace e(s)，初始化为 0
        self.e = np.zeros(n_states)

        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam

    def reset_trace(self):
        """
        在每个 episode 开始时调用
        将所有 eligibility trace 清零
        """
        self.e[:] = 0.0

    def update(self, s, r, s_next, done):
        """
        执行一次 TD(λ) 更新

        参数：
        s       : 当前状态
        r       : 当前奖励
        s_next  : 下一个状态
        done    : episode 是否结束
        """

        # -----------------------------
        # 1. 计算 TD 误差 δ
        # -----------------------------
        if done:
            # 如果是终止状态，没有下一个状态价值
            td_target = r
        else:
            td_target = r + self.gamma * self.V[s_next]

        td_error = td_target - self.V[s]

        # -----------------------------
        # 2. 更新 eligibility trace
        # -----------------------------
        # 当前访问状态的 trace +1
        self.e[s] += 1.0

        # -----------------------------
        # 3. 对所有状态执行 TD(λ) 更新
        # -----------------------------
        for state in range(len(self.V)):
            # 价值函数更新
            self.V[state] += self.alpha * td_error * self.e[state]

            # eligibility trace 衰减
            self.e[state] *= self.gamma * self.lam

    def get_value(self, s):
        """返回状态 s 的当前价值估计"""
        return self.V[s]
```

* * *

七、训练循环示例（伪环境）
-------------

```python
def train(agent, env, n_episodes=1000):
    for episode in range(n_episodes):
        s = env.reset()
        agent.reset_trace()

        done = False
        while not done:
            # 这里使用随机策略（仅为示例）
            action = random.choice(env.actions)

            s_next, r, done = env.step(action)

            # TD(λ) 更新
            agent.update(s, r, s_next, done)

            s = s_next
```

* * *

八、工程实践中的关键问题
------------

### 1\. 状态空间过大？

*   使用 **函数逼近版本：TD(λ) + NN**
*   eligibility trace 推广为 **参数级 trace**

### 2\. λ 过大不稳定？

*   λ ↑ → 偏向 MC → 方差变大
*   实际工程常用：**0.7 ~ 0.9**

### 3\. 与策略梯度的关系？

*   Actor-Critic 中：
    *   Critic 常用 **TD(λ)**
    *   例如：A3C、GAE(λ)

* * *

九、总结一句话版本
---------

> **TD(λ) = 用 eligibility trace 将一步 TD 误差沿时间轴“软反向传播”**

如果你愿意，下一步我可以继续为你讲解：

*   TD(λ) 的 **前向视角（λ-return）**
*   TD(λ) 在 **函数逼近下的严格形式**
*   **GAE(λ)** 与 TD(λ) 的数学等价关系
*   TD(λ) 在 **深度强化学习中的工程实现**

你可以直接指定下一步想深入的方向。

## Prompt:
继续讲解【使用重要性采样的多步离线学习法】

## Response:
下面我将继续以**强化学习领域教师**的角色，系统讲解你提出的主题：

> **使用重要性采样（Importance Sampling）的多步离线学习方法（Off-policy Multi-step Learning）**

这一主题是 **TD(λ)、n-step TD、Off-policy 学习** 的交汇点，也是理解 **Retrace、V-trace、Q(λ)** 等现代算法的基础。

* * *

一、问题背景：为什么需要“重要性采样的多步离线学习”
==========================

1\. 在线 vs 离线（On-policy vs Off-policy）
-------------------------------------

在强化学习中，我们区分两种策略：

*   **行为策略（Behavior Policy）**：  
    用来采样数据的策略，记为  $\mu(a|s)$ 
*   **目标策略（Target Policy）**：  
    我们真正想评估或优化的策略，记为  $\pi(a|s)$ 

### Off-policy 学习的核心问题

> **数据来自策略 μ，但我们想学习策略 π**

* * *

2\. 为什么多步方法在 Off-policy 下变得困难？
------------------------------

### On-policy n-step return

$$
G_t^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
$$

在 **On-policy** 情况下是无偏的。

* * *

### Off-policy 问题

如果轨迹由  $\mu$  生成，而我们假设它来自  $\pi$ ，则：

*   多步回报 **不再是目标策略下的期望**
*   会引入 **系统性偏差**

解决方案：**重要性采样**

* * *

二、重要性采样（Importance Sampling）基础
==============================

1\. 核心思想（概率论角度）
---------------

对任意函数  $f$ ：

$$
\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right]
$$

* * *

2\. 强化学习中的重要性采样比率
-----------------

在一个时间步：

$$
\rho_t = \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}
$$

在多步轨迹中：

$$
\rho_{t:t+n-1} = \prod_{k=t}^{t+n-1} \frac{\pi(a_k|s_k)}{\mu(a_k|s_k)}
$$

* * *

3\. 一个关键直觉
----------

*   如果某个动作在目标策略中 **概率更高**  
    → 放大该样本的贡献
*   如果在目标策略中 **几乎不会发生**  
    → 样本权重趋近 0

* * *

三、使用重要性采样的 n-step 离线 TD 学习
==========================

1\. Off-policy n-step TD 目标
---------------------------

$$
G_t^{(n, IS)} = \rho_{t:t+n-1} \left( r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n}) \right)
$$

* * *

2\. 离线 TD 更新形式
--------------

$$
V(s_t) \leftarrow V(s_t) + \alpha \left( G_t^{(n, IS)} - V(s_t) \right)
$$

* * *

3\. 严重问题：方差爆炸
-------------

重要性采样比率是**连乘**：

*   稍有偏差
*   多步后指数级放大
*   实际工程中极不稳定

* * *

四、加权重要性采样（Weighted IS）
======================

1\. 思想
------

用归一化权重减少方差：

$$
\bar{\rho}_t = \frac{\rho_t}{\sum \rho}
$$

但代价是：

*   **引入偏差**
*   但一致性仍然成立

* * *

2\. 工程取舍
--------

| 方法 | 偏差 | 方差 | 稳定性 |
| --- | --- | --- | --- |
| 普通 IS | 无偏 | 极高 | 差 |
| 加权 IS | 有偏 | 低 | 好 |

* * *

五、从 n-step IS 到 TD(λ)（Off-policy）
=================================

1\. Off-policy TD(λ) 的本质
------------------------

> **对所有 n-step 回报进行 λ 加权平均 + 重要性采样修正**

$$
G_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \rho_{t:t+n-1} G_t^{(n)}
$$

* * *

2\. 后向视角：Eligibility Trace + IS
-------------------------------

这直接导致 **Off-policy Eligibility Trace**：

$$
e_t(s) = \rho_t \left( \gamma \lambda e_{t-1}(s) + \mathbb{I}[s=s_t] \right)
$$

* * *

六、使用重要性采样的离线 TD(λ) 算法
=====================

1\. TD 误差（不变）
-------------

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

* * *

2\. 更新规则（关键）
------------

$$
V(s) \leftarrow V(s) + \alpha \delta_t e_t(s)
$$

其中：

$$
e_t(s) = \rho_t \left( \gamma \lambda e_{t-1}(s) + \mathbb{I}[s=s_t] \right)
$$

* * *

3\. 一个重要结论（理论上）
---------------

> **Off-policy TD(λ) 在函数逼近下可能发散**

这直接推动了后续算法的发展。

* * *

七、教学伪代码（Off-policy TD(λ) + IS）
==============================

```text
初始化 V(s), e(s) = 0

for each episode:
    s ← s0
    e(s) ← 0

    while not terminal:
        从行为策略 μ 采样动作 a
        执行动作，得到 r, s'

        ρ = π(a|s) / μ(a|s)
        δ = r + γ V(s') - V(s)

        for all states s_i:
            e(s_i) = ρ * (γ λ e(s_i))
        e(s) += ρ

        for all states s_i:
            V(s_i) += α δ e(s_i)

        s ← s'
```

* * *

八、Python 实现（离散状态，教学版）
=====================

```python
import numpy as np

class OffPolicyTDLambda:
    def __init__(self, n_states, alpha=0.1, gamma=0.99, lam=0.8):
        """
        使用重要性采样的离线 TD(λ)

        n_states : 状态数量
        alpha    : 学习率
        gamma    : 折扣因子
        lam      : λ 参数
        """

        self.V = np.zeros(n_states)
        self.e = np.zeros(n_states)

        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam

    def reset_trace(self):
        """每个 episode 开始时清空 eligibility trace"""
        self.e[:] = 0.0

    def update(self, s, a, r, s_next, done, pi, mu):
        """
        执行一次 Off-policy TD(λ) 更新

        参数：
        s      : 当前状态
        a      : 当前动作
        r      : 奖励
        s_next : 下一个状态
        done   : 是否终止
        pi     : 目标策略函数 pi(a|s)
        mu     : 行为策略函数 mu(a|s)
        """

        # -----------------------------
        # 1. 计算重要性采样比率
        # -----------------------------
        rho = pi(a, s) / mu(a, s)

        # -----------------------------
        # 2. 计算 TD 误差
        # -----------------------------
        if done:
            td_target = r
        else:
            td_target = r + self.gamma * self.V[s_next]

        delta = td_target - self.V[s]

        # -----------------------------
        # 3. 更新 eligibility trace
        # -----------------------------
        # 对所有状态进行衰减并乘以 rho
        self.e *= self.gamma * self.lam * rho

        # 当前状态额外增加 rho
        self.e[s] += rho

        # -----------------------------
        # 4. 更新价值函数
        # -----------------------------
        self.V += self.alpha * delta * self.e
```

* * *

九、工程与研究中的重要延伸
=============

1\. 为什么要裁剪重要性权重？
----------------

*   原始 IS 方差过大
*   实际系统几乎不可用

### 裁剪形式

$$
\bar{\rho}_t = \min(c, \rho_t)
$$

* * *

2\. 现代算法的演化路径
-------------

| 算法 | 核心思想 |
| --- | --- |
| Q(λ) | Off-policy TD(λ) |
| Retrace(λ) | 截断 IS，保证收敛 |
| V-trace | Actor-Critic 稳定训练 |
| GAE(λ) | On-policy 低方差优势估计 |

* * *

十、一句话总结
-------

> **使用重要性采样的多步离线学习，本质是在“纠正策略不匹配”的同时，尝试保留多步回报的低偏差优势，但代价是方差与稳定性问题。**

* * *

如果你愿意，下一步我可以进一步讲解：

1.  **Retrace(λ) 的严格推导与收敛性保证**
2.  **V-trace 在 IMPALA 中的工程实现**
3.  **Off-policy TD(λ) 为什么在函数逼近下会发散（Baird 反例）**
4.  **GAE(λ) 与重要性采样的关系澄清**

你可以直接指定下一讲的方向。

## Prompt:
继续讲解【多步树回湖算法】

## Response:
下面我将继续以**强化学习教师**的视角，系统讲解你提出的主题：

> **多步树回溯算法（Multi-step Tree-backup Algorithm）**

这是一个**非常关键但容易被忽略**的算法族成员。它位于以下方法之间的“中间地带”：

*   n-step TD（不修正 off-policy 偏差）
*   重要性采样多步 TD（无偏但高方差）
*   Retrace / V-trace（工程化折中）

**Tree-backup 的核心价值在于：  
👉 它是一种“不使用重要性采样、但仍然是 off-policy 的多步算法”**

* * *

一、提出 Tree-backup 的动机
====================

1\. 回顾：off-policy 多步学习的两难
-------------------------

| 方法 | 是否 off-policy | 偏差 | 方差 |
| --- | --- | --- | --- |
| n-step TD | ❌ | 大 | 低 |
| IS n-step TD | ✅ | 无 | 极高 |
| TD(λ)+IS | ✅ | 小 | 仍然很高 |

**核心问题**：

> 能否在 off-policy 下，避免重要性采样的连乘爆炸？

* * *

2\. Tree-backup 的基本思想（一句话）
--------------------------

> **不沿着行为策略实际执行的动作“回传”，  
> 而是沿着目标策略的“期望动作树”进行回传**

这就是“Tree-backup”名称的来源。

* * *

二、从一步期望备份（Expected SARSA）说起
===========================

Tree-backup 可以看作是 **Expected SARSA 的多步推广**。

1\. 一步 Expected SARSA
---------------------

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_t + \gamma \sum_a \pi(a|s_{t+1}) Q(s_{t+1}, a) - Q(s_t, a_t) \right)
$$

注意：

*   不使用重要性采样
*   在  $s_{t+1}$  处对 **所有动作取期望**

* * *

2\. 核心特征
--------

*   **对未来分支“平均”**
*   不关心行为策略 μ
*   天然是 off-policy 的

* * *

三、Tree-backup 的核心思想（直觉层面）
=========================

1\. 与 n-step SARSA 的对比
----------------------

### n-step SARSA

*   沿着 **单条采样轨迹**
*   未来动作完全由 μ 决定

### Tree-backup

*   只有 **第一步** 使用真实采样动作
*   之后的动作全部：
    
$$
\text{用 } \sum_a \pi(a|s) Q(s,a)
$$

> 从时间 t+1 开始，“展开一棵动作树”，但用期望值折叠

* * *

2\. 为什么不需要重要性采样？
----------------

因为：

*   Tree-backup **从定义上就在目标策略 π 下**
*   并未假设后续动作来自 μ

* * *

四、n-step Tree-backup 的数学定义
==========================

下面以 **动作价值函数 Q(s,a)** 为例（Tree-backup 通常用于 Q-learning）。

* * *

1\. 定义辅助量：期望动作价值
----------------

$$
\bar{Q}(s) = \sum_a \pi(a|s) Q(s,a)
$$

* * *

2\. n-step Tree-backup 目标（递归形式）
-------------------------------

定义回报  $G_t^{(n)}$ ：

### 终止条件

$$
G_{t+n}^{(0)} = \bar{Q}(s_{t+n})
$$

### 递归定义

$$
G_t^{(n)} = r_t + \gamma \left[ \sum_{a \neq a_t} \pi(a|s_{t+1}) Q(s_{t+1}, a) + \pi(a_t|s_{t+1}) G_{t+1}^{(n-1)} \right]
$$

* * *

3\. 非常重要的直觉
-----------

*   **只对“采样到的动作”继续展开**
*   其他动作直接用当前 Q 估计
*   所以是一棵“被折叠的树”

* * *

五、Tree-backup(λ)：多步 + 资格迹形式
===========================

Tree-backup 同样可以写成 **λ-return 的形式**。

* * *

1\. Tree-backup 的 TD 误差
-----------------------

$$
\delta_t = r_t + \gamma \bar{Q}(s_{t+1}) - Q(s_t, a_t)
$$

* * *

2\. Eligibility Trace（Tree-backup 特有）
-------------------------------------

$$
e_t(s,a) = \gamma \lambda \pi(a_t|s_t) e_{t-1}(s,a) + \mathbb{I}[(s,a) = (s_t,a_t)]
$$

注意：

*   **trace 衰减时乘的是 π，不是 ρ**
*   这是与 Off-policy TD(λ) 的本质区别

* * *

3\. 参数更新
--------

$$
Q \leftarrow Q + \alpha \delta_t e_t
$$

* * *

六、Tree-backup(λ) 教学伪代码
======================

```text
初始化 Q(s,a), e(s,a) = 0

for each episode:
    初始化 s
    e(s,a) = 0

    while not terminal:
        用行为策略 μ 采样 a
        执行动作，得到 r, s'

        计算:
        Q_bar = Σ_a π(a|s') Q(s',a)
        δ = r + γ Q_bar - Q(s,a)

        对所有 (s_i,a_i):
            e(s_i,a_i) *= γ λ π(a|s)

        e(s,a) += 1

        对所有 (s_i,a_i):
            Q(s_i,a_i) += α δ e(s_i,a_i)

        s ← s'
```

* * *

七、Python 实现（教学版，离散动作空间）
=======================

```python
import numpy as np

class TreeBackupLambda:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, lam=0.8):
        """
        Tree-backup(λ) 算法实现（教学版本）

        n_states  : 状态数量
        n_actions : 动作数量
        """

        self.Q = np.zeros((n_states, n_actions))
        self.e = np.zeros((n_states, n_actions))

        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.n_actions = n_actions

    def reset_trace(self):
        """每个 episode 开始时清空 eligibility trace"""
        self.e[:] = 0.0

    def expected_Q(self, s, pi):
        """
        计算目标策略 π 在状态 s 下的期望 Q 值
        """
        return sum(pi(a, s) * self.Q[s, a] for a in range(self.n_actions))

    def update(self, s, a, r, s_next, done, pi):
        """
        执行一次 Tree-backup(λ) 更新
        """

        # 1. 计算期望 Q
        if done:
            q_bar = 0.0
        else:
            q_bar = self.expected_Q(s_next, pi)

        # 2. TD 误差
        delta = r + self.gamma * q_bar - self.Q[s, a]

        # 3. eligibility trace 衰减（乘 π）
        self.e *= self.gamma * self.lam * pi(a, s)

        # 4. 当前 (s,a) 的 trace 增加
        self.e[s, a] += 1.0

        # 5. 更新 Q
        self.Q += self.alpha * delta * self.e
```

* * *

八、Tree-backup 与其他算法的系统对比
========================

| 方法 | 是否 IS | Off-policy | 方差 | 偏差 |
| --- | --- | --- | --- | --- |
| SARSA(λ) | ❌ | ❌ | 低 | 高 |
| IS TD(λ) | ✅ | ✅ | 极高 | 无 |
| Tree-backup(λ) | ❌ | ✅ | 低 | 小 |
| Retrace(λ) | 截断 | ✅ | 低 | 极小 |

* * *

九、Tree-backup 的工程意义
===================

1.  **理论上优雅**
2.  **避免 IS 的不稳定**
3.  是 Retrace 的重要理论前身
4.  在小规模离散问题中表现良好

但：

*   需要已知 **目标策略 π**
*   连续动作空间下计算期望代价高

* * *

十、一句话总结
=======

> **Tree-backup 是一种通过“对未来动作取期望”来规避重要性采样、实现稳定 off-policy 多步学习的方法。**

* * *

如果你愿意，下一步我可以继续为你深入讲解：

1.  **Retrace(λ) 如何统一 IS 与 Tree-backup**
2.  **Tree-backup 与 Expected SARSA 的严格关系**
3.  **为什么 Tree-backup 在深度 RL 中较少直接使用**
4.  **多步方法的统一视角（Sutton 2018 λ-operator）**

你可以直接指定下一讲的方向。

## Prompt:
对比下面多种算法，并解释各个算法之间的关系
- SARSA算法
- Q learning 算法
- TD(λ)算法
- 使用重要性采样的多步离线学习法
- 多步树回湖算法
- <其他类似的算法？若有，请列出并对比>

## Response:
下面我将以**强化学习课程中“算法谱系梳理”这一讲**的方式，对你列出的算法进行**系统对比、统一建模与关系解释**。目标不是孤立介绍算法，而是让你**在一个坐标系中理解它们为什么存在、解决什么问题、彼此如何演化**。

* * *

一、先给你一张“总览地图”（结论先行）
===================

可以用 **三条核心维度** 来统一理解这些算法：

> **① 是否 on-policy / off-policy  
> ② 是否单步 / 多步  
> ③ 如何处理 off-policy 的偏差–方差问题**

* * *

算法在三维空间中的位置（先直观）
----------------

| 算法 | On / Off-policy | 单步 / 多步 | 核心技术 |
| --- | --- | --- | --- |
| SARSA | On-policy | 单步 | 采样下一动作 |
| Q-learning | Off-policy | 单步 | max 备份 |
| TD(λ) | On-policy | 多步 | Eligibility Trace |
| IS 多步离线学习 | Off-policy | 多步 | Importance Sampling |
| Tree-backup | Off-policy | 多步 | 期望备份（无 IS） |
| Retrace(λ) | Off-policy | 多步 | 截断 IS + Tree-backup |
| Expected SARSA | Off-policy | 单步 | 动作期望 |
| n-step SARSA | On-policy | 多步 | 多步采样回报 |

下面我们**逐层展开**。

* * *

二、第一层：SARSA 与 Q-learning —— 一切的起点
=================================

* * *

1\. SARSA：最“朴素”的 TD 控制算法
------------------------

### 更新公式

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \big[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t) \big]
$$

### 核心特征

*   **On-policy**
*   使用“实际执行的下一个动作”
*   学的是 **当前行为策略的 Q**

### 直觉

> “我怎么走，就学我怎么走”

* * *

2\. Q-learning：off-policy 的经典代表
-------------------------------

### 更新公式

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \big[ r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t,a_t) \big]
$$

### 核心特征

*   **Off-policy**
*   行为策略可以是 ε-greedy
*   学的是 **贪婪目标策略**

### 直觉

> “我乱走，但我假装未来一定最优”

* * *

3\. SARSA vs Q-learning（本质区别）
-----------------------------

| 维度 | SARSA | Q-learning |
| --- | --- | --- |
| 策略类型 | On-policy | Off-policy |
| 备份方式 | 采样动作 | max |
| 风险敏感性 | 高（更保守） | 低（更激进） |
| 工程稳定性 | 好 | 易过估计 |

> **Expected SARSA 是它们的“中间态”**

* * *

三、第二层：TD(λ) —— 从“单步”到“多步”
=========================

* * *

1\. TD(λ) 的本质
-------------

> **TD(λ) 不是一个新目标，而是“如何使用历史 TD 误差”的机制**

### 核心机制：Eligibility Trace

$$
Q \leftarrow Q + \alpha \delta_t e_t
$$

### λ 的作用

*   λ = 0 → 单步 TD
*   λ → 1 → Monte Carlo
*   在偏差–方差之间连续插值

* * *

2\. TD(λ) 与 SARSA / Q-learning 的关系
----------------------------------

| 基础算法 | 加上 λ 后 |
| --- | --- |
| SARSA | SARSA(λ) |
| Q-learning | Q(λ)（理论复杂） |

⚠️ 关键点：  
**标准 TD(λ) 默认是 on-policy 的**

* * *

四、第三层：off-policy + 多步 = 灾难的开始
=============================

* * *

1\. 为什么 off-policy 多步很难？
------------------------

多步回报假设：

> “未来动作来自目标策略 π”

但现实是：

*   数据来自 μ
*   策略不匹配
*   偏差累积

* * *

2\. 使用重要性采样的多步离线学习法
-------------------

### 核心修正手段

$$
\rho_t = \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}
$$
 
$$
G_t^{(n)} = \rho_{t:t+n-1} \cdot \text{n-step return}
$$

### 优缺点

| 优点 | 缺点 |
| --- | --- |
| 理论无偏 | 方差指数级爆炸 |
| 原理清晰 | 工程不可用 |

> **这是“理论正确但实践失败”的典型代表**

* * *

3\. Off-policy TD(λ) + IS
-------------------------

$$
e_t \leftarrow \rho_t (\gamma \lambda e_{t-1} + 1)
$$

⚠️ 已知问题：

*   在函数逼近下可能发散（Baird 反例）

* * *

五、第四层：Tree-backup —— 放弃 IS 的聪明选择
================================

* * *

1\. Tree-backup 的核心思想
---------------------

> **不沿采样轨迹回传，而是对未来动作取期望**

### TD 目标

$$
r + \gamma \sum_a \pi(a|s') Q(s',a)
$$

* * *

2\. Tree-backup 的地位
-------------------

| 特性 | Tree-backup |
| --- | --- |
| 是否 IS | 否 |
| 是否 off-policy | 是 |
| 方差 | 低 |
| 偏差 | 小 |

### 与 IS 方法的根本差异

*   IS：修正“走错路”
*   Tree-backup：**从一开始就不假设你走对**

* * *

3\. Tree-backup(λ)
------------------

Eligibility trace 更新为：

$$
e_t \leftarrow \gamma \lambda \pi(a_t|s_t) e_{t-1}
$$

而不是乘 ρ。

* * *

六、第五层：统一视角 —— 这些算法到底是什么关系？
==========================

* * *

1\. 一条清晰的演化主线
-------------

```text
SARSA
  ↓ 多步
SARSA(λ)
  ↓ off-policy
IS 多步 TD
  ↓ 稳定性改造
Tree-backup
  ↓ 理论 + 工程统一
Retrace(λ)
  ↓ 大规模并行
V-trace
```

* * *

2\. Retrace(λ)：Tree-backup + 截断 IS
----------------------------------

$$
c_t = \min(1, \rho_t)
$$
 
$$
e_t \leftarrow \gamma \lambda c_t e_{t-1} + 1
$$

✔ 收敛  
✔ 低方差  
✔ 工程可用

* * *

七、其他相关算法（你要求的扩展）
================

| 算法 | 关系定位 |
| --- | --- |
| Expected SARSA | 单步 Tree-backup |
| n-step SARSA | On-policy 多步 |
| Q(λ) | Q-learning + trace（不稳定） |
| GAE(λ) | TD(λ) 在策略梯度中的形式 |
| V-trace | Retrace 的 Actor-Critic 版本 |
| Double Q-learning | 解决 max 偏置 |
| Munos λ-operator | 统一理论框架 |

* * *

八、一张“终极对照表”
===========

| 算法 | On/Off | 多步 | IS | 稳定性 |
| --- | --- | --- | --- | --- |
| SARSA | On | ❌ | ❌ | ⭐⭐⭐⭐ |
| Q-learning | Off | ❌ | ❌ | ⭐⭐⭐ |
| TD(λ) | On | ✅ | ❌ | ⭐⭐⭐⭐ |
| IS 多步 TD | Off | ✅ | ✅ | ⭐ |
| Tree-backup | Off | ✅ | ❌ | ⭐⭐⭐⭐ |
| Retrace(λ) | Off | ✅ | 截断 | ⭐⭐⭐⭐⭐ |
| V-trace | Off | ✅ | 截断 | ⭐⭐⭐⭐⭐ |

* * *

九、一句话总总结（非常重要）
==============

> **强化学习中多步算法的发展史，本质是一部“如何在 off-policy 场景下控制方差而不引入不可接受偏差”的历史。**

* * *

如果你愿意，下一步我可以进一步帮你做三件事之一：

1.  **用一张数学统一公式推导所有这些算法**
2.  **结合 DQN / Actor-Critic，解释现代算法到底在用哪一个**
3.  **从工程角度告诉你：真实系统中应该选谁、为什么**

你可以直接指定方向。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)