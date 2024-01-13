---
layout: default
usemathjax: true
---

# How I implemented and tuned Discrete Soft Actor Critic

**Caution!** This is not a peer-review conference paper, just a blog post. Therefore take the findings with a pinch of salt 
abd verify! 

## Automatic entropy tuning in SAC
The authors of [1] proposed the following optimization problem 

$$
\max_{\pi(\cdot)} ~\mathbb{E}_{\rho_\pi} [r(s_t, a_t)], 
\text{ s.t. } ~\mathbb{E}_{(s_t,a_t) \sim \rho_\pi} \left[ \log(\pi(a_t|s_t)) | \right] \geq H_{\rm target},
$$

where $$H_{\rm target}$$ is the target entropy of the optimal policy. This problem can be solved using a Lagrangian approach:

$$
\max_{\pi(\cdot)}\min_{\alpha \ge 0} ~\mathbb{E}_{\rho_\pi} [r(s_t, a_t)] + \alpha \mathbb{E}_{(s_t,a_t) \sim \rho_\pi} \left(-\log(\pi(a_t|s_t)) - H_{\rm target} \right).
$$

The algorithm then updates the policy $$\pi$$, the coefficient $$\alpha$$ during the learning procedure.

In the case, of Gaussian action distribution centred around zero, the entropy can be computed as 

$$H_{\rm target} = 0.5 * \log(\sigma^2 * 2 * \pi * e),$$ 

which is equal to $$-0.88$$ for $$\sigma=0.1$$. This means that the lower bound on the policy entropy should be a Gaussian with a small variance.

## How to set the target entropy in the discrete SAC?
### Standard approach
How do these ideas translate to the discrete case? In [2] it proposed to 
set the target entropy as follows:

```python
    target_entropy = - 0.98 * np.log(1.0 / action_space.n)
```

What does it mean for a policy with two actions? The target entropy is now $0.67$, while the entropy for a uniform distribution (with $p=0.5$ for both actions) is equal to $0.69$! 

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
where $\pi(s)$ is the greedy (deterministic) policy, which is chosen with probability $\varepsilon$. While the uniformly random policy is chosen with probability $1-\varepsilon$. Computing the entropy of this policy can be hard, but we appoximate it using the following expression:

```python
    epsilon = 0.95
    # log probability of the greedy action                
    target_entropy =-epsilon * np.log(epsilon)                  
                   # log probability of random actions
                   +(1-epsilon) * np.log((1-epsilon)/(action_space.n-1))
```
Thefore, we can ensure that our stochastic policy is at least as stochastic as an epsilon-greedy policy. 

For a two-action policy the target entropy is now equal to $0.19$,
which is closer to zero indicating a more determinstic approach. For $0.99$ the target entropy becomes $0.05$.

## Experiments

While the derivation may seem plausible, the proof is always in the pudding or [rather in eating it](https://en.wiktionary.org/wiki/the_proof_of_the_pudding_is_in_the_eating#:~:text=The%20current%20phrasing%20is%20generally,you%20fry%20the%20eggs%E2%80%9D).


## Literature

[1] Haarnoja T, Zhou A, Hartikainen K, Tucker G, Ha S, Tan J, Kumar V, Zhu H, Gupta A, Abbeel P, Levine S. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905. 2018 Dec 13.

[2] Christodoulou, Petros. “Soft actor-critic for discrete action settings.” arXiv preprint arXiv:1910.07207 (2019).
