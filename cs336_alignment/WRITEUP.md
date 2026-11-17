# My Write-up for Assignment 5: Alignment

## Problem (prompting_baseline)

### (a)
Each prompting approach was evaluated on 1,319 questions. The table reports the number of responses in each category.

| Prompting approach | Format = 1, correctness = 1 | Format = 1, correctness = 0 | Format = 0, correctness = 0 |
| --- | ---: | ---: | ---: |
| Question-only | 0 | 219 | 1,100 |
| Zero-shot | 0 | 792 | 527 |
| Few-shot | 217 | 1,046 | 56 |

It is surprising that few-shot prompting improves correctness this much. When I reviewed responses that were formatted correctly but received an answer reward of 0, I noticed repeated reasoning patterns that were related to mathematical problems but unrelated to the specific question.

### (b)

The results above suggest that this model benefits from examples that demonstrate the task and help it apply the same approach to new questions. This may differ for larger models. Both few-shot and zero-shot prompting improved formatting. They also encouraged the model to generate reasoning, which may have helped it solve problems it could not solve with a short answer alone. Providing examples may have helped the model draw on mathematical knowledge acquired during pretraining. The examples may also have made the task more familiar, reducing the irrelevant or incoherent text I observed with question-only prompting.

## Problem (baseline_calcs)

### (a)

Notation:

- $A_i$: the action sampled on trial $i$, either $0$ or $1$.
- $n$: the number of sampled actions.
- $\theta$: the model's trainable parameter (a single number in this problem).
- $\pi_\theta$: the policy, or probability distribution over actions.
- $\pi_\theta(A_i)$: the probability assigned to the sampled action $A_i$.
- $\sigma$: the sigmoid function, $\sigma(\theta)=\frac{1}{1+e^{-\theta}}$.
- $p$: the probability of action $1$; action $0$ has probability $1-p$.
- $r(A_i)$: the reward, equal to $1$ for action $1$ and $0$ for action $0$.
- $\log$: the natural logarithm.
- $\nabla_\theta$: differentiation with respect to $\theta$.
- $Z_i$: one sampled action's contribution to the gradient estimate.
- $\hat g$: the estimate of the expected-reward gradient with respect to $\theta$, averaged over $n$ samples.
- $\mathbb{E}$: expectation, or the probability-weighted average.
- $\operatorname{Var}$: variance, the expected squared distance from the expectation.
- $\mathrm{i.i.d.}$: independent and identically distributed; every action is sampled independently using the same fixed policy.
- $\sim$: sampled from a distribution.

$$
A_1,\ldots,A_n \overset{\mathrm{i.i.d.}}{\sim} \pi_\theta,
\qquad p=\sigma(\theta).
$$

$$
Z_i=r(A_i)\nabla_\theta\log\pi_\theta(A_i),
\qquad
\hat g=\frac{1}{n}\sum_{i=1}^{n}Z_i.
$$

$$
\nabla_\theta\log\pi_\theta(1)
=\nabla_\theta\log p
=\nabla_\theta\log\sigma(\theta)
=\frac{1}{\sigma(\theta)}\sigma(\theta)(1-\sigma(\theta))
=1-p.
$$

$$
Z_i=
\begin{cases}
0 & A_i=0 \quad (\text{probability }1-p),\\
1-p & A_i=1 \quad (\text{probability }p).
\end{cases}
$$

$$
\mathbb{E}[Z_i]=(1-p)\cdot 0+p(1-p)=p-p^2=p(1-p).
$$

$$
\mathbb{E}[Z_i^2]=(1-p)\cdot 0^2+p(1-p)^2=p(p-1)^2.
$$

$$
\operatorname{Var}(Z_i)
=\mathbb{E}[Z_i^2]-(\mathbb{E}[Z_i])^2
=p(1-p)^2-[p(1-p)]^2
=p(1-p)^3.
$$

$$
\operatorname{Var}(\hat g)
=\frac{\operatorname{Var}(Z_i)}{n}
=\frac{p(1-p)^3}{n}.
$$

### (b)

$b$: a constant reward baseline.

$$
Z_i=(r(A_i)-b)\nabla_\theta\log\pi_\theta(A_i),
\qquad
\hat g=\frac{1}{n}\sum_{i=1}^{n}Z_i.
$$

$$
\nabla_\theta\log\pi_\theta(0)
=\nabla_\theta\log(1-p)
=-\frac{p(1-p)}{1-p}
=-p.
$$

$$
Z_i=
\begin{cases}
bp & A_i=0 \quad (\text{probability }1-p),\\
(1-b)(1-p) & A_i=1 \quad (\text{probability }p).
\end{cases}
$$

$$
\mathbb{E}[Z_i]
=(1-p)bp+p(1-b)(1-p)
=p(1-p).
$$

$$
\mathbb{E}[Z_i^2]
=(1-p)b^2p^2+p(1-b)^2(1-p)^2.
$$

$$
\operatorname{Var}(Z_i)
=\mathbb{E}[Z_i^2]-(\mathbb{E}[Z_i])^2
$$

$$
=(1-p)b^2p^2+p(1-b)^2(1-p)^2-p^2(1-p)^2
$$

$$
=p(1-p)\left[b^2p+(1-b)^2(1-p)-p(1-p)\right]
=p(1-p)(1-p-b)^2.
$$

$$
\operatorname{Var}(\hat g)
=\frac{\operatorname{Var}(Z_i)}{n}
=\frac{p(1-p)(1-p-b)^2}{n}.
$$

### (c)

$$
b=p,
\qquad
\operatorname{Var}(\hat g)
=\frac{p(1-p)(1-2p)^2}{n}.
$$

Compared with the variance without a baseline, $\frac{p(1-p)^3}{n}$, this variance can be lower, equal, or higher depending on $p$. Using the population mean as the baseline does not always reduce variance.
