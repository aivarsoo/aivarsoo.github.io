---
layout: default
description: Notes on discrete SAC
usemathjax: true
---


**Caution!** This is not a peer-review conference paper, just a blog post. Therefore take the findings with a pinch of salt 
abd verify! 

## Automatic entropy tuning in SAC
The authors of [1] proposed the following optimization problem 

$$
\max_{\pi(\cdot)} ~\mathbb{E}_{(s_t,a_t) \sim \rho_\pi} [r(s_t, a_t)], 
\text{ s.t. } ~\mathbb{E}_{(s_t,a_t) \sim \rho_\pi} \left[ \log(\pi(a_t|s_t)) | \right] \geq H_{\rm target},
$$

where $$H_{\rm target}$$ is the target entropy of the optimal policy. This problem can be solved using a Lagrangian approach:

$$
\max_{\pi(\cdot)}\min_{\alpha \ge 0} ~\mathbb{E}_{(s_t,a_t) \sim \rho_\pi}[r(s_t, a_t)] + \alpha \mathbb{E}_{(s_t,a_t) \sim \rho_\pi} \left(-\log(\pi(a_t|s_t)) - H_{\rm target} \right).
$$

The algorithm then updates the policy $$\pi$$, the coefficient $$\alpha$$ during the learning procedure.

In the case, of Gaussian action distribution centred around zero, the entropy can be computed as 

$$H_{\rm target} = 0.5 * \log(\sigma^2 * 2 * \pi * e),$$ 

which is equal to $$-0.88$$ for $$\sigma=0.1$$. This means that the lower bound on the policy entropy should be a Gaussian with a small variance.

## Implementation of entropy tuning
Given the min-max problem above the implementation of entropy tuning is performed by updating $$\alpha$$ using the gradient descent and derivating 
the objective, i.e.:

$$
\alpha := \alpha - l_{r, \alpha} \nabla_{\alpha} (\mathbb{E}_{(s_t,a_t) \sim \rho_\pi} [r(s_t, a_t)] + \alpha \mathbb{E}_{(s_t,a_t) \sim \rho_\pi} \left(-\log(\pi(a_t|s_t)) - H_{\rm target} \right))
$$

where $$l_{r,\alpha}$$ is the learning rate of the parameter $$\alpha$$. Taking the gardient we end up with the following update:

$$
\alpha := \alpha + l_{r, \alpha} \mathbb{E}_{(s_t,a_t) \sim \rho_\pi} \left(\log(\pi(a_t|s_t)) + H_{\rm target} \right)
$$

In practice, however, things differ ever so slightly. First, of all to restrict $$\alpha$$ to be positive one typically optimizes over $$\log(\alpha)$$ instead. Leading to the following objective:

$$
\mathbb{E}_{(s_t,a_t) \sim \rho_\pi} [r(s_t, a_t)] + \exp(\beta) \mathbb{E}_{(s_t,a_t) \sim \rho_\pi} \left(-\log(\pi(a_t|s_t)) - H_{\rm target} \right)
$$

where $$\beta = \log(\alpha)$$. However, in practice the gradient update is different! For example, in the paper [Revisiting Discrete Soft Actor Critic](https://arxiv.org/abs/2209.10081) the authors used the following loss in [their implementation](https://github.com/coldsummerday/Revisiting-Discrete-SAC/blob/main/src/libs/discrete_sac.py#L231):

```python
    log_prob = - entropy.detach() + self._target_entropy
    alpha_loss = -(self._log_alpha * log_prob).mean()
```

The same loss is used in [Ray](https://github.com/ray-project/ray/blob/master/rllib/algorithms/sac/sac_torch_policy.py#L319C18-L319C18) and [TorchRL](https://github.com/pytorch/rl/blob/main/torchrl/objectives/sac.py#L761). From the mathematical perspective it seems correct to impelement 
the following loss instead:

```python
    log_prob = - entropy.detach() + self._target_entropy
    alpha_loss = -(torch.exp(self._log_alpha) * log_prob).mean()
```

While the standard loss is mathematically incorrect, it does result in gradients of a correct direction. The only difference is the norm of the gradients, which is especially noticeable for values of $$\alpha$$ deviating from $$1$$. So why the standard loss is chosen this way? It is argued that this update was in the original paper and therefore should be used unless there is evidence that "the correct loss" leads to improved performance. I haven't seen such evidence and perhaps this is the reason why this loss is still popular. 

## How to set the target entropy in the discrete SAC?
### Standard approach
How do these ideas translate to the discrete case? In [2] it proposed to 
set the target entropy as follows:

```python
    target_entropy = - 0.98 * np.log(1.0 / action_space.n)
```

What does it mean for a policy with two actions? The target entropy is now $$0.67$$, while the entropy for a uniform distribution (with a probability $$p=0.5$$ for both actions) is equal to $$0.69$$! 

This means that the target entropy of the policy should be (almost) equal to *the entropy of the random policy*. This does not feel right...

<!-- If I understood correctly the authors of [1], the target entropy is meant to be the lower bound on the policy entropy. Here we are effectively trying to set the lower bound of the policy entropy to the entropy of the uniform distribution, which maximises the entropy. In my experiments, this results in \alpha growing with every iteration and makes the performance of the algorithm unstable. -->

### A novel (?) approach 
In the discrete action case the classical approach is using an epislon-greedy approach that is picking an action according to the rule
$$
    a = 
    \begin{cases}
        \pi(s)  & \epsilon, \\
        \text{random} & 1-\epsilon,
    \end{cases}
$$
where $$\pi(s)$$ is the greedy (deterministic) policy, which is chosen with probability $$\varepsilon$$. While the uniformly random policy is chosen with probability $$1-\varepsilon$$. Computing the entropy of this policy can be hard, but we appoximate it using the following expression:

```python
    epsilon = 0.95
    # log probability of the greedy action                
    target_entropy =-epsilon * np.log(epsilon)                  
                   # log probability of random actions
                   +(1-epsilon) * np.log((1-epsilon)/(action_space.n-1))
```
Thefore, we can ensure that our stochastic policy is at least as stochastic as an epsilon-greedy policy. 

For a two-action policy the target entropy is now equal to $$0.19$$,
which is closer to zero indicating a more determinstic approach. For $$0.99$$ the target entropy becomes $$0.05$$.

## Experiments

While the derivation may seem plausible, the proof is always in the pudding or [rather in eating it](https://en.wiktionary.org/wiki/the_proof_of_the_pudding_is_in_the_eating#:~:text=The%20current%20phrasing%20is%20generally,you%20fry%20the%20eggs%E2%80%9D).


## Literature

[1] Haarnoja T, Zhou A, Hartikainen K, Tucker G, Ha S, Tan J, Kumar V, Zhu H, Gupta A, Abbeel P, Levine S. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905. 2018 Dec 13.

[2] Christodoulou, Petros. “Soft actor-critic for discrete action settings.” arXiv preprint arXiv:1910.07207 (2019).
