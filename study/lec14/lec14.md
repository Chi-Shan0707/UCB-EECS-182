# Imitation Learning

From "learning to predict" to "learning to control"

Independent and Identically Distributed

IID : $p(\mathcal{D}) = \prod_i p(y_i|x_i)p(x_i)$

each decision can change future inputs(not independent)
supervision may be high-level
objective is to accomplish the task



use a nn to fit  专家行为？

会有问题

Non-Markovian beahvior??

Multimodal behavior


考虑pi_theta(a_t|o_t)是马尔科夫性的
人类有记忆 之前看见告示牌 后面会减速

同时人类同一个observation会有多种行为
比如左边绕或右边绕
就多峰




- 解决问题1 就是RNN
