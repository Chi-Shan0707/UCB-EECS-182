结合前后两页讲义的内容，这两页PPT完整地构建了深度学习中**从概率视角（Probabilistic Perspective）推导确定性损失函数（Deterministic Loss Functions）**的理论框架。

讲义的核心目的在于证明：我们在神经网络回归任务中广泛使用的启发式目标——**均方误差（Mean Squared Error, MSE）**，在严格的统计学意义上，等价于对**具有恒定方差的高斯分布进行最大似然估计（Maximum Likelihood Estimation, MLE）**。

需要严肃指正的是，第二页PPT在符号表达上存在一定的不严谨。PPT中写道：“Also the same as the mean squared error (MSE) loss! $\mathcal{L}(\theta, x, y) = -\frac{1}{2} ||f_\theta(x) - y||^2$”。严格来说，**损失函数（Loss Function）是需要被最小化的**，因此 MSE 损失应当是正值（即去掉负号）。PPT此处的 $\mathcal{L}$ 实际上是指代需要被**最大化**的对数似然（Log-Likelihood），而非严格意义上需要被**最小化**的损失误差。

以下是基于这两页讲义的完整、深度的 Markdown 结构化笔记。

---

# 深度学习误差分析理论基础：最大似然估计与均方误差的等价性

## 1. 范式转换：从标量预测到分布预测 (Paradigm Shift)

在现代机器学习的严谨框架中，神经网络 $f_\theta(x)$ 并非直接输出一个确定性的标量或类别，而是输出一个**概率分布的参数**。这种视角的转换是理解诸如交叉熵（Cross-Entropy）和均方误差（MSE）等损失函数本质的前提。

| 任务类型 | 传统确定性视角 (Deterministic View) | 现代概率视角 (Probabilistic View) | 模型输出的具体物理意义 |
| :--- | :--- | :--- | :--- |
| **分类 (Classification)** | 输出确定的离散标签（如：狗） | 输出离散概率分布 | 类别概率（如：$[0.1, 0.9, 0.0]$），对应Categorical分布的参数 |
| **回归 (Regression)** | 输出确定的连续数值（如：15.2） | 输出连续概率密度分布 | 分布的统计参数（如高斯分布的均值 $\mu$ 和协方差 $\Sigma$） |

## 2. 连续分布的数学推导：以高斯分布为例

在回归任务中，我们假设给定输入 $x$ 时，目标变量 $y$ 的真实条件概率分布服从多元高斯分布（Multivariate Gaussian Distribution）。

神经网络扮演的角色是拟合该分布的参数。假设网络输出即为分布的均值 $\mu_\theta(x) = f_\theta(x)$，而协方差矩阵为 $\Sigma_\theta(x)$。

### 2.1 构建条件对数似然 (Conditional Log-Likelihood)

多元高斯分布的概率密度函数（PDF）定义为：
$$p_\theta(y|x) = \frac{1}{\sqrt{(2\pi)^k |\Sigma_\theta(x)|}} \exp\left( -\frac{1}{2}(f_\theta(x) - y)^T \Sigma_\theta(x)^{-1} (f_\theta(x) - y) \right)$$

为方便优化，我们对其取自然对数，得到对数似然函数（Log-Likelihood）：
$$\log p_\theta(y|x) = -\frac{1}{2}(f_\theta(x) - y)^T \Sigma_\theta(x)^{-1} (f_\theta(x) - y) - \frac{1}{2}\log|\Sigma_\theta(x)| - \frac{k}{2}\log(2\pi)$$

将常数项 $-\frac{k}{2}\log(2\pi)$ 记作 $\text{const}$，即得到第一页讲义中的核心方程：
$$\log p_\theta(y|x) = -\frac{1}{2}(f_\theta(x) - y)^T \Sigma_\theta(x)^{-1} (f_\theta(x) - y) - \frac{1}{2}\log|\Sigma_\theta(x)| + \text{const}$$

* **物理意义**：第一项本质上是预测值与真实值之间的**马哈拉诺比斯距离（Mahalanobis Distance）**的平方。它表明，误差的惩罚应当由协方差矩阵 $\Sigma_\theta(x)$ 进行加权（即方差越大的维度，对误差的容忍度越高）。

### 2.2 简化假设：各向同性与同方差性

为了建立与标准 MSE 的联系，我们引入一个极其强烈的统计学假设：**假设模型的预测误差是同方差的（Homoscedastic），且各维度相互独立。**

在数学上，这表现为协方差矩阵 $\Sigma_\theta(x)$ 是一个单位矩阵 $\mathbf{I}$。
代入 $\Sigma_\theta(x) = \mathbf{I}$：
1.  逆矩阵 $\mathbf{I}^{-1} = \mathbf{I}$。
2.  行列式 $|\mathbf{I}| = 1$，因此 $\log|\mathbf{I}| = 0$。
3.  二次型退化为 L2 范数（欧几里得距离）的平方：$(f_\theta(x) - y)^T \mathbf{I} (f_\theta(x) - y) = ||f_\theta(x) - y||^2$。

对数似然方程因此大幅简化为：
$$\log p_\theta(y|x) = -\frac{1}{2} ||f_\theta(x) - y||^2 + \text{const}$$

## 3. 统计推断与损失函数的统一 (MLE $\equiv$ MSE)

根据**最大似然估计（Maximum Likelihood Estimation, MLE）**原则，模型训练的目标是寻找一组参数 $\theta$，使得观测数据出现的概率最大化。即我们需要**最大化** $\log p_\theta(y|x)$。

由于常数项 $\text{const}$ 不影响参数 $\theta$ 的梯度，优化目标等价于最大化：
$$\mathcal{L}_{MLE}(\theta) = -\frac{1}{2} ||f_\theta(x) - y||^2$$

### 3.1 纠正讲义中的术语表达

正如第二页 PPT 所示，讲义将 $-\frac{1}{2} ||f_\theta(x) - y||^2$ 称为“Loss”。这是一个表述上的瑕疵。
在优化理论中，“损失（Loss）”或“代价（Cost）”是被定义为需要被**最小化**的正向惩罚项。
最大化对数似然（MLE），在数学上完全等价于**最小化负对数似然（Negative Log-Likelihood, NLL）**。

因此，严格的 MSE 损失函数 $J(\theta)$ 应当是 $\mathcal{L}_{MLE}$ 取负号：
$$J(\theta) = - \log p_\theta(y|x) \propto \frac{1}{2} ||f_\theta(x) - y||^2$$

### 3.2 结论总结

这两页讲义提供了一个至关重要的理论证明：
**当我们使用均方误差（MSE）作为神经网络的损失函数时，我们实际上是在潜意识地做一个强烈的假设——即数据的残差服从一个均值为 0、方差为常数 $\mathbf{I}$ 的高斯分布。**

这种底层逻辑的理解对于后续研究至关重要。例如，如果面对异方差（Heteroscedastic）数据（即噪声随输入 $x$ 改变），直接使用标准的 MSE 将导致次优解。此时，依据上述公式的第一步推导，让网络同时输出均值 $f_\theta(x)$ 和对角协方差 $\sigma_\theta^2(x)$，并直接最小化完整的负对数似然，才是更为科学的算法设计路径。

需要我为您进一步展开讲解在 $\Sigma_\theta(x) \neq \mathbf{I}$ 时，如何通过修改网络结构和损失函数来建模认知不确定性（Aleatoric Uncertainty）吗？