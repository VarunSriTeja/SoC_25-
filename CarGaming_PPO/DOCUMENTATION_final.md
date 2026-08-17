# CarRacing-v3 PPO Project --- Technical Documentation & Interview Study Guide

## 1. Project Overview

This project studies **Proximal Policy Optimization (PPO)** on
Gymnasium's `CarRacing-v3`, a continuous-control reinforcement-learning
environment in which an agent must learn to steer, accelerate, and brake
from raw visual observations.

Two implementations were developed and compared:

1.  **Scratch PPO** --- PPO implemented manually in PyTorch, including
    the CNN actor-critic, Gaussian policy, rollout collection,
    Generalized Advantage Estimation (GAE), clipped PPO objective,
    entropy regularization, value-function learning, parallel
    environments, learning-rate scheduling, checkpointing, and
    evaluation.
2.  **Stable-Baselines3 PPO (SB3)** --- a mature library implementation
    used as a baseline.

The final comparison used the same fixed 10 evaluation tracks.

  -----------------------------------------------------------------------
  Metric                            Scratch PPO                   SB3 PPO
  ------------------- ------------------------- -------------------------
  Selected training        200,704 interactions    \~300,000 interactions
  checkpoint                                    

  Mean evaluation                     **156.5**                 **220.5**
  reward                                        

  Standard deviation                      165.3                     173.6

  Best evaluation                     **584.4**                     547.5
  episode                                       

  Worst evaluation                         -2.7                  **44.4**
  episode                                       
  -----------------------------------------------------------------------

The key result is not that the custom implementation beats SB3 overall.
SB3 achieved the better mean and consistency, while the Scratch PPO
demonstrated that a manually implemented PPO agent could learn
meaningful visual driving behavior and even obtain the highest
individual evaluation score.

------------------------------------------------------------------------

# Part I --- Understanding the Reinforcement-Learning Problem

## 2. What is CarRacing-v3?

`CarRacing-v3` is a Gymnasium Box2D environment. At every environment
step the agent receives an RGB image of the racing scene and chooses a
three-dimensional continuous action:

-   **Steering:** approximately `[-1, 1]`
-   **Gas:** `[0, 1]`
-   **Brake:** `[0, 1]`

The agent receives reward primarily for visiting track tiles and is
penalized over time. Therefore, a successful policy must learn not
merely to move forward but to remain on the track and cover new road
efficiently.

This is substantially harder than a small state-vector control problem
because the policy must learn both:

1.  **visual representation** --- extract useful road/track information
    from pixels;
2.  **continuous control** --- map those learned features into steering,
    throttle, and braking decisions.

That is why a convolutional actor-critic architecture is appropriate.

------------------------------------------------------------------------

## 3. Reinforcement-Learning Formulation

At time step (t):

-   state/observation: (s_t)
-   action: (a_t)
-   reward: (r_t)
-   next state: (s\_{t+1})

The objective is to learn policy (`\pi`\_`\theta`(a\|s))
that maximizes expected discounted return:

$$
 G_t = \sum*{k=0}^{\infty}\gamma^k r*{t+k} 
$$

where (`\gamma`) is the discount factor.

In this project:

$$
 \gamma = 0.99 
$$

A value function estimates:

$$
 V\_\phi(s_t) \approx \mathbb{E}$$
G_t\|s_t
$$

$$

The policy answers **what action should I take?**

The critic answers **how good is this state?**

PPO trains both.

------------------------------------------------------------------------

# Part II --- Final Scratch PPO Implementation

## 4. Why PPO?

PPO is an **on-policy actor-critic policy-gradient algorithm**.

A naive policy-gradient method can make updates that are too large. A single destructive update may radically change the policy and destroy previously learned behavior.

PPO addresses this by limiting how much the new policy is rewarded for moving away from the policy that collected the training data.

The central PPO probability ratio is:

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$

The clipped objective is:

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta) A_t,
\operatorname{clip}
\left(
r_t(\theta),
1-\epsilon,
1+\epsilon
\right)
A_t
\right)
\right]
$$

with:

$$
\epsilon = 0.2
$$

### Interview explanation

A concise explanation is:

> PPO is a policy-gradient algorithm that improves an actor using advantage estimates while restricting overly aggressive policy updates through a clipped probability-ratio objective. It provides a practical balance between training stability and implementation simplicity.

------------------------------------------------------------------------

# Part III --- Observation Processing

## 5. Why Raw RGB Frames Were Not Used Directly

The original environment produces approximately `96 x 96 x 3` RGB
observations.

Feeding these directly would:

-   increase computation,
-   contain irrelevant dashboard/border pixels,
-   make the network learn redundant color information,
-   provide no temporal information from a single frame.

The final pipeline therefore performs:

``` text
96×96 RGB frame
        ↓
crop
        ↓
84×84
        ↓
grayscale
        ↓
normalize to [-1, 1]
        ↓
stack 4 consecutive frames
        ↓
(4, 84, 84)
```

------------------------------------------------------------------------

## 6. Cropping

The preprocessing uses:

``` python
frame = frame[:-12, 6:-6]
```

This converts the `96 x 96` frame to `84 x 84`.

Why crop rather than arbitrarily resize the complete image?

Cropping removes less useful border/dashboard content while retaining
the important racing region. It also avoids unnecessarily distorting the
geometry of the road.

------------------------------------------------------------------------

## 7. Grayscale Conversion

RGB is converted to grayscale using approximately:

$$
 Y = 0.299R + 0.587G + 0.114B 
$$

Why?

The primary driving problem depends strongly on:

-   road boundaries,
-   curvature,
-   relative position,
-   upcoming track geometry.

A grayscale representation reduces the input from three channels to one
while preserving much of the spatial information required for control.

------------------------------------------------------------------------

## 8. Normalization

Pixel values are transformed from `[0,255]` to `[-1,1]`.

Conceptually:

$$
 x' = 2\left(\frac{x}{255}\right) - 1 
$$

Neural networks generally optimize more effectively when inputs have
controlled numerical scales rather than raw values as large as 255.

------------------------------------------------------------------------

## 9. Four-Frame Stacking

A single image tells the agent where the car is but provides weak
information about motion.

Four consecutive grayscale frames are stacked:

``` text
frame t-3
frame t-2
frame t-1
frame t
```

giving:

``` text
(4, 84, 84)
```

This allows the CNN to infer temporal information indirectly, including:

-   movement direction,
-   rate of rotation,
-   apparent velocity,
-   how quickly road geometry is approaching.

This is a lightweight alternative to introducing an RNN/LSTM.

### Interview question: Why not use one frame?

Because the environment is partially observable from a single image with
respect to velocity and motion. Frame stacking supplies short-term
temporal context without the complexity of a recurrent architecture.

------------------------------------------------------------------------

# Part IV --- CNN Actor-Critic

## 10. Why a CNN?

Images have strong spatial structure. A fully connected network would
ignore locality and require many more parameters.

The Scratch PPO feature extractor uses convolutional layers
approximately following:

``` text
Input: 4 × 84 × 84

Conv2D
16 filters
8 × 8 kernel
stride 4
LeakyReLU

↓

Conv2D
32 filters
3 × 3 kernel
stride 2
LeakyReLU

↓

Flatten
```

The convolutional representation is then used by:

-   the **actor** to construct an action distribution;
-   the **critic** to predict state value.

------------------------------------------------------------------------

## 11. Actor-Critic Architecture

The shared visual features branch into two objectives.

``` text
                CNN
                 │
          learned features
             /       \
            /         \
        Actor         Critic
          │              │
   action distribution   V(s)
```

### Actor

The actor parameterizes:

$$
 \pi\_\theta(a\|s) 
$$

### Critic

The critic estimates:

$$
 V\_\phi(s) 
$$

The critic does not directly choose an action. It provides a baseline
for estimating whether an action outcome was better or worse than
expected.

------------------------------------------------------------------------

# Part V --- Continuous Gaussian Policy

## 12. Why a Probability Distribution Instead of Direct Actions?

Policy-gradient algorithms require a stochastic policy during training.

Rather than output:

``` text
steering = 0.2
gas = 0.6
brake = 0.0
```

the actor defines a distribution around possible actions.

For continuous control, the implementation uses Gaussian distributions:

$$
 a
\sim \mathcal{N}(\mu\_\theta(s),\sigma)

$$

The network predicts action means and learns action standard deviations.

The initial standard deviation was approximately:

$$
 \sigma = 0.4 
$$

for each action dimension.

------------------------------------------------------------------------

## 13. Exploration vs Exploitation

A larger standard deviation means greater exploration.

A smaller standard deviation makes the policy more deterministic.

During training, sampling from the distribution allows the agent to
explore different steering/throttle/braking behavior.

During deterministic evaluation, the distribution mean is used:

$$
 a = \mu\_\theta(s) 
$$

This removes sampling noise and gives a repeatable evaluation policy.

------------------------------------------------------------------------

## 14. Action Mapping

The network's bounded mean representation is mapped to the actual
CarRacing action ranges:

``` text
steering → [-1, 1]
gas      → [0, 1]
brake    → [0, 1]
```

This is important because the action dimensions do not all have the same
physical bounds.

------------------------------------------------------------------------

# Part VI --- Parallel Environment Collection

## 15. Why 16 Parallel Environments?

The final Scratch implementation uses:

``` text
num_envs = 16
horizon = 128
```

Therefore each rollout update collects:

$$
 16 \times 128 = 2048 
$$

environment transitions.

Parallel environments provide:

-   greater data throughput,
-   more diverse tracks/trajectories per update,
-   reduced correlation compared with one continuous trajectory,
-   larger PPO batches.

A consequence seen in the logs is that many episodes finished in groups
of roughly 16. Therefore `avg10` remained unchanged across several
training updates until another group of environments completed episodes.

------------------------------------------------------------------------

## 16. Rollout Buffer

For each horizon, the algorithm stores information such as:

-   observations,
-   sampled actions,
-   rewards,
-   terminal flags,
-   value predictions.

Conceptually:

``` text
for t in rollout:
    observe state
    sample action
    execute action
    receive reward
    store transition
```

After 128 steps per environment, PPO stops collecting data temporarily
and performs optimization.

This is why PPO is **on-policy**: the training data comes from the
current policy and is used for a limited set of updates before new data
is collected.

------------------------------------------------------------------------

# Part VII --- Generalized Advantage Estimation

## 17. What is an Advantage?

The advantage measures whether an action performed better than expected
from a state.

Conceptually:

$$
 A_t = Q(s_t,a_t)-V(s_t) 
$$

Positive advantage:

> This action produced a better result than expected.

Negative advantage:

> This action produced a worse result than expected.

The policy should increase the probability of positive-advantage actions
and decrease the probability of negative-advantage actions.

------------------------------------------------------------------------

## 18. Temporal-Difference Residual

GAE starts with:

$$
 \delta*t = r_t + \gamma V(s*{t+1}) - V(s_t) 
$$

with terminal masking where appropriate.

------------------------------------------------------------------------

## 19. GAE

Generalized Advantage Estimation combines TD residuals:

$$
 A_t\^{GAE} = \delta*t +
(\gamma\lambda)\delta*{t+1} +
(\gamma\lambda)\^2\delta\_{t+2} +\dots

$$

The implementation uses:

$$
 \gamma = 0.99 
$$

$$
 \lambda = 0.95 
$$

### Why GAE?

Policy-gradient estimation has a bias-variance tradeoff.

Using only immediate TD errors can introduce bias.

Using complete Monte Carlo returns can have high variance.

GAE provides a practical compromise.

### Interview explanation

> GAE computes exponentially weighted multi-step TD residuals. Lambda
> controls the bias-variance tradeoff; values near one use longer-term
> information, while lower values rely more strongly on short-horizon
> value estimates.

------------------------------------------------------------------------

# Part VIII --- Advantage Normalization

## 20. Why Normalize Advantages?

The collected advantages are standardized approximately as:

$$
 \hat A = \frac{A-\mu_A}
{\sigma\_A+\epsilon} 
$$

This makes optimization less sensitive to the absolute numerical scale
of returns and generally produces more stable gradient magnitudes.

------------------------------------------------------------------------

# Part IX --- Frozen Old Policy

## 21. Why Have `policy` and `old_policy`?

Before PPO optimization, the current network is copied:

``` text
old_policy ← policy
```

The old policy is frozen during the optimization epochs.

Then:

``` text
old_policy → probability under data-collection policy
policy     → probability under updated policy
```

This allows calculation of:

$$
 r_t(\theta) = \exp[
\log\pi_\theta(a_t|s_t)
-
\log\pi_{\theta_{old}}(a_t|s_t)
]
$$

The old policy provides the fixed reference required by PPO's clipped
objective.

------------------------------------------------------------------------

# Part X — PPO Clipping

## 22. Understanding the Probability Ratio

Suppose an action was likely under the old policy and becomes much more likely under the new policy.

Then:

$$
r_t > 1
$$

If it becomes less likely:

$$
r_t < 1
$$

Without constraints, the optimizer might push this ratio too far.

PPO clips the probability ratio in the surrogate objective to approximately:

$$
[0.8,\;1.2]
$$

because:

$$
\epsilon = 0.2
$$

This does not literally prevent every parameter change outside this interval. Instead, it removes the incentive from the clipped surrogate objective for excessively large beneficial changes in the probability ratio.

------------------------------------------------------------------------

## 23. Policy Loss

The policy maximizes:

$$
 \min( r_t A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
) 
$$

The implementation minimizes the negative of this surrogate objective.

------------------------------------------------------------------------

# Part XI --- Critic / Value Loss

## 24. Value Function Training

The critic learns to predict the return target.

A mean-squared-error style loss is used:

$$
 L_V = (V(s_t)-R_t)\^2 
$$

with value coefficient:

$$
 c_v=0.5 
$$

The critic is essential because GAE depends on value predictions.

A weak critic produces noisy or misleading advantage estimates.

------------------------------------------------------------------------

# Part XII --- Entropy Regularization

## 25. What is Entropy?

Entropy measures uncertainty in the policy distribution.

For a Gaussian policy, higher standard deviation generally corresponds
to greater entropy.

The total PPO objective includes an entropy bonus.

The coefficient used is:

$$
 c\_{entropy}=0.01 
$$

Why?

Without exploration pressure, the policy can become deterministic too
early and get trapped in poor behavior.

The entropy term encourages continued exploration during learning.

------------------------------------------------------------------------

# Part XIII --- Complete PPO Objective

## 26. Combined Loss

Conceptually, optimization combines:

$$
 L = -L\_{policy} + c_vL\_{value} - c_eH(\pi) 
$$

where:

-   (L\_{policy}): clipped PPO surrogate;
-   (L\_{value}): critic regression loss;
-   (H(`\pi`)): entropy;
-   (c_v=0.5);
-   (c_e=0.01).

The optimizer is Adam.

------------------------------------------------------------------------

# Part XIV --- PPO Optimization Schedule

## 27. Multiple Epochs

Each collected rollout is reused for:

``` text
10 PPO epochs
```

with:

``` text
minibatch size = 128
```

Why reuse data?

Environment interaction is expensive. PPO performs several controlled
optimization passes over each on-policy rollout.

Why not unlimited epochs?

Too many passes cause the policy to move too far from the behavior
policy, making the data increasingly off-policy and risking destructive
updates.

------------------------------------------------------------------------

## 28. Learning Rate

Initial learning rate:

$$
 3\times10\^{-4} 
$$

The final implementation uses a decay schedule of the form:

$$
 LR = LR_0 \times
0.85\^{\lfloor step/10000\rfloor} 
$$

where the relevant step is the optimizer-step schedule used by the
implementation.

During the successful run, the logs showed transitions such as:

``` text
3.00e-4
→ 2.55e-4
→ 2.17e-4
```

Lowering the learning rate later in training allows more conservative
refinement.

------------------------------------------------------------------------

# Part XV --- Training Diagnostics

## 29. What Did the Logs Mean?

Typical logs contained:

``` text
steps
episodes
avg10
loss
pi
v
ent
kl
clip
lr
act
```

### `steps`

Total environment interactions across all parallel environments.

### `episodes`

Number of completed episodes.

### `avg10`

Mean reward over the most recently completed 10 episodes.

It stayed identical across several logs when no new episodes completed.

### `loss`

Combined optimization objective.

A loss value alone is not a reliable measure of RL performance.
Reward/evaluation matters more.

### `pi`

Policy surrogate statistic/loss component.

### `v`

Value loss component.

### `ent`

Entropy contribution/statistic.

### `kl`

Approximate KL divergence between old and updated policy.

Large KL values can indicate an aggressive policy update.

### `clip`

Fraction of samples for which the probability ratio moved outside the
PPO clipping region.

A very high clip fraction means many updates are hitting the clipping
boundary.

### `act`

Mean sampled steering, gas and brake actions for the rollout.

This was useful for detecting degenerate policies such as excessive
throttle, collapsed braking, or persistent steering bias.

------------------------------------------------------------------------

# Part XVI --- Why KL Was Monitored

## 30. Approximate KL Divergence

KL divergence gives a rough measure of how far the new policy has moved
from the old one.

PPO clipping reduces destructive updates but does not guarantee tiny KL
divergence.

During experimentation, occasional KL spikes appeared.

These spikes helped explain why reinforcement-learning performance could
deteriorate even after a period of strong rewards.

Important interview point:

> PPO clipping improves stability but does not guarantee monotonic
> improvement. Neural-network optimization, finite sampling,
> value-function error, and repeated minibatch updates can still produce
> unstable policy changes.

------------------------------------------------------------------------

# Part XVII --- Checkpointing

## 31. Why Save Intermediate Models?

RL training is not monotonic.

The latest model is not necessarily the best model.

This project demonstrated that clearly.

Scratch checkpoint evaluation:

  Checkpoint          Mean         Std        Best      Worst
  ------------ ----------- ----------- ----------- ----------
  151k                53.8        99.1       250.2      -55.6
  **200k**       **156.5**   **165.3**   **584.4**   **-2.7**
  251k                67.2       102.2       268.9      -31.1

The 200k checkpoint substantially outperformed the later 251k
checkpoint.

Therefore:

> Never assume the final RL checkpoint is the strongest checkpoint.

Checkpointing also protected training from Colab runtime interruptions.

------------------------------------------------------------------------

# Part XVIII --- The Successful Scratch Training Run

## 32. Learning Progression

The successful second Scratch run showed a clear transition:

``` text
~69k   avg10 ≈ -10.3
~82k   avg10 ≈  -8.5
~96k   avg10 ≈  -2.4
~113k  avg10 ≈  -0.2
~129k  avg10 ≈ +24.8
~145k  avg10 ≈ +104.6
~162k  avg10 ≈ +242.0
~193k  avg10 ≈ +323.7
```

This was strong evidence that the policy had learned meaningful
track-following behavior.

Later performance declined:

``` text
~209k → 266.5
~225k → 189.1
~242k → 151.2
~258k → 173.4
~272k → 108.9
```

This motivated explicit checkpoint evaluation rather than simply taking
the final policy.

------------------------------------------------------------------------

# Part XIX --- Fixed Evaluation

## 33. Why Training Reward Was Not Enough

Training reward is collected:

-   while actions are stochastic,
-   on training trajectories,
-   while the policy is changing.

It is not a clean generalization metric.

Therefore models were evaluated on the same fixed set of 10 tracks/seeds
using deterministic actions.

For Scratch 200k:

``` text
3.7
183.8
50.5
259.7
-2.7
21.4
584.4
143.4
174.4
145.9
```

giving:

$$
 156.5 \pm 165.3 
$$

with best:

$$
 584.4 
$$

and worst:

$$
 -2.7 
$$

------------------------------------------------------------------------

# Part XX --- Qualitative Scratch Behavior

## 34. What the Gameplay Demonstrated

The selected Scratch demo showed:

-   successful handling of multiple corners,
-   two particularly clean early turns,
-   loss of the ideal line on a later turn,
-   recovery back toward the track,
-   successful navigation of additional smaller turns,
-   eventual difficulty on a larger/sharper corner.

This is important because reward alone does not describe behavior.

The video provides qualitative evidence that the CNN learned meaningful
visual-control features rather than merely producing arbitrary actions.

------------------------------------------------------------------------

# Part XXI --- Earlier Experiments and How the Final Approach Was Reached

## 35. Initial Scratch PPO Attempts

Before the final implementation, simpler Scratch PPO versions were
explored.

Problems included:

-   poor reward progression,
-   policies that missed corners,
-   circling behavior after leaving the track,
-   steering/action imbalance,
-   aggressive throttle behavior,
-   policy instability during continuation.

These experiments were useful because they showed that simply
implementing the PPO equations was not sufficient; preprocessing, action
modeling, rollout organization, hyperparameters, and training stability
all mattered.

------------------------------------------------------------------------

## 36. Stable Continuation Experiment

A continuation strategy lowered the learning rate and introduced more
conservative PPO behavior.

It reduced some extreme policy-update instability, but reward
performance still deteriorated and action behavior remained poor.

Lesson:

> Stable optimization metrics do not necessarily imply a useful policy.

A model can have reasonable KL divergence while learning behavior that
is bad for the task.

------------------------------------------------------------------------

## 37. Beta Action Distribution Experiment

A bounded Beta distribution was tested because steering/gas/brake have
bounded ranges.

The initialization successfully produced sensible initial action
statistics such as approximately:

``` text
steering ≈ 0
gas ≈ 0.4
brake ≈ 0.13
```

and numerical diagnostics were healthy.

However, the 25k training gate deteriorated strongly in reward.

Lesson:

> Fixing action bounds alone did not solve the fundamental learning
> problem.

This hypothesis was therefore rejected rather than wasting additional
training compute.

------------------------------------------------------------------------

## 38. Environment/Reward Experiment

An off-track heuristic was briefly explored to terminate unproductive
behavior.

The first heuristic incorrectly detected green pixels around the road as
off-track and caused unrealistically short episodes.

This was identified from the episode count: hundreds of episodes were
finishing in a tiny number of total environment steps.

Lesson:

> Environment shaping must be validated carefully. A seemingly
> reasonable heuristic can fundamentally change the MDP and invalidate
> training results.

That experiment was discarded.

------------------------------------------------------------------------

## 39. Final Direction

The project then moved to a more disciplined implementation based on a
known CarRacing PPO design while modernizing it for:

-   PyTorch,
-   Gymnasium,
-   `CarRacing-v3`,
-   current Colab environments.

The final implementation preserved the important PPO mechanics and
CarRacing preprocessing while avoiding speculative modifications.

A second random seed then produced the successful learning trajectory.

------------------------------------------------------------------------

# Part XXII --- Seed Sensitivity

## 40. Why Did Run 2 Perform Better?

Reinforcement learning is stochastic.

Randomness enters through:

-   network initialization,
-   stochastic action sampling,
-   randomly generated tracks,
-   minibatch shuffling,
-   parallel environment scheduling,
-   GPU numerical behavior.

The first full run peaked weakly and later degraded.

The second clean run, using a different seed, reached a training rolling
reward above 300 and produced the final selected model.

This is not unusual in deep RL.

### Interview point

Never report one RL run as proof of universal performance. Seed
sensitivity should be acknowledged, and stronger studies normally
evaluate multiple seeds.

For this project, multiple runs were used primarily as an engineering
diagnostic and to demonstrate reproducibility challenges.

------------------------------------------------------------------------

# Part XXIII --- Stable-Baselines3 Baseline

## 41. Why Use SB3?

A from-scratch implementation tells us whether we understand PPO.

A library baseline tells us how our implementation compares with a
mature, widely used implementation.

Stable-Baselines3 provides:

-   tested PPO update logic,
-   robust rollout handling,
-   optimized vector-environment support,
-   standardized policy implementations,
-   logging/checkpoint utilities.

It therefore serves as an engineering benchmark.

------------------------------------------------------------------------

## 42. SB3 Preprocessing

The SB3 agent used compatible visual preprocessing/frame stacking so
that the comparison remained focused on PPO implementation quality
rather than radically different observation information.

The environment was wrapped into a vectorized form and image dimensions
were transposed as required by SB3's CNN pipeline.

------------------------------------------------------------------------

## 43. SB3 Training

SB3 PPO was trained for approximately 300k interactions, including
checkpoint/resume handling.

Representative final training diagnostics included:

``` text
ep_len_mean ≈ 995
ep_rew_mean ≈ 376
explained_variance ≈ 0.927
```

These are training statistics rather than the final fixed-track
evaluation.

The final model was separately evaluated.

------------------------------------------------------------------------

# Part XXIV --- SB3 Evaluation

## 44. Fixed 10-Track Results

SB3 evaluation scores:

``` text
44.4
203.6
82.8
486.3
47.7
57.1
547.5
146.3
282.7
306.6
```

Mean:

$$
 220.5 
$$

Standard deviation:

$$
 173.6 
$$

Best:

$$
 547.5 
$$

Worst:

$$
 44.4 
$$

Unlike Scratch, all ten SB3 evaluation episodes were positive.

This supports the conclusion that SB3 was more consistent overall.

------------------------------------------------------------------------

# Part XXV --- Scratch vs SB3

## 45. Final Comparison

  Property                Scratch PPO      SB3 PPO
  ----------------------- ---------------- ----------------------------
  PPO implementation      Manual PyTorch   Library
  CNN                     Custom           SB3 CNN pipeline
  GAE                     Manual           Built in
  PPO clipping            Manual           Built in
  Rollout management      Manual           Built in
  Parallel environments   16               SB3 vector env
  Checkpoint selection    Explicit         Final/baseline checkpoints
  Selected steps          \~200k           \~300k
  Mean                    156.5            **220.5**
  Best                    **584.4**        547.5
  Worst                   -2.7             **44.4**

### Interpretation

SB3 wins on:

-   mean reward,
-   worst-case reward,
-   overall consistency.

Scratch demonstrates:

-   successful manual PPO implementation,
-   meaningful visual-control learning,
-   competitive behavior on several tracks,
-   highest individual evaluation reward.

A fair conclusion is:

> The custom PPO learned meaningful CarRacing behavior and occasionally
> matched or exceeded the mature baseline, but SB3 remained more robust
> and consistent across unseen tracks.

------------------------------------------------------------------------

# Part XXVI --- Important Interview Questions

## 46. What Makes PPO Different from Vanilla Policy Gradient?

Vanilla policy gradient can make very large destructive updates.

PPO uses the probability ratio between the new and old policies and
clips the optimization objective to discourage excessive updates.

------------------------------------------------------------------------

## 47. Why is PPO On-Policy?

The rollout was generated by the current behavior policy.

After several PPO epochs, that data is discarded and fresh data is
collected.

Reusing old experience indefinitely would make it increasingly
off-policy.

------------------------------------------------------------------------

## 48. Why Do We Need the Critic?

The critic estimates expected return from a state.

It provides a baseline for calculating advantages, reducing
policy-gradient variance and enabling GAE.

------------------------------------------------------------------------

## 49. Why GAE Instead of Plain Returns?

GAE balances bias and variance using exponentially weighted TD
residuals.

It is usually more stable than raw Monte Carlo return estimates.

------------------------------------------------------------------------

## 50. Why Normalize Advantages?

It standardizes the scale of the policy-gradient signal, improving
optimization stability across batches.

------------------------------------------------------------------------

## 51. Why Use Entropy?

Entropy discourages premature collapse to a deterministic policy and
maintains exploration.

------------------------------------------------------------------------

## 52. Why Use Frame Stacking?

A single frame poorly represents motion.

Four frames provide short-term temporal context such as direction and
apparent velocity without requiring a recurrent network.

------------------------------------------------------------------------

## 53. Why Use a CNN?

The observation is an image. CNNs exploit spatial locality and parameter
sharing, making them more appropriate and efficient than fully connected
layers for raw visual observations.

------------------------------------------------------------------------

## 54. Why Use Parallel Environments?

They increase throughput and trajectory diversity and reduce correlation
within each rollout batch.

------------------------------------------------------------------------

## 55. Why Can Reward Decline After Improving?

Deep RL optimization is non-stationary and stochastic.

Possible reasons include:

-   destructive policy updates,
-   critic errors,
-   sampling variance,
-   exploration changes,
-   over-specialization to recent trajectories,
-   large KL shifts.

This project directly observed non-monotonic performance.

------------------------------------------------------------------------

## 56. Why Was the 200k Model Selected Instead of 251k?

Because models were evaluated on the same fixed 10 tracks.

The 200k checkpoint achieved:

$$
 156.5 
$$

mean reward versus only:

$$
 67.2 
$$

for 251k.

The latest checkpoint was therefore not the best-generalizing
checkpoint.

------------------------------------------------------------------------

## 57. Why is SB3 Better on Average?

SB3 is a mature implementation with extensively tested:

-   numerical details,
-   rollout handling,
-   optimization behavior,
-   policy architecture,
-   defaults and training utilities.

A custom implementation is educational and flexible but easier to
destabilize.

------------------------------------------------------------------------

## 58. Why Didn't Scratch Beat SB3 Overall?

That was not necessary for the project to succeed.

The purpose of Scratch PPO was to demonstrate algorithmic understanding
and build a working agent.

SB3's higher mean confirms why mature RL libraries are valuable in
production/research workflows.

------------------------------------------------------------------------

# Part XXVII --- Engineering Lessons

## 59. RL Debugging Requires More Than Reward

Useful diagnostics included:

-   episode reward,
-   rolling reward,
-   KL divergence,
-   clip fraction,
-   entropy,
-   action means,
-   episode counts,
-   gameplay videos.

For example, unrealistic episode counts exposed a faulty termination
heuristic immediately.

------------------------------------------------------------------------

## 60. Video Evaluation Matters

A scalar reward cannot show:

-   whether the car actually follows corners,
-   whether it spins,
-   whether it recovers after leaving the road,
-   whether high reward comes from sensible driving.

Gameplay videos were therefore an important qualitative validation tool.

------------------------------------------------------------------------

## 61. Checkpointing is Essential in Colab

Long-running RL jobs can be interrupted.

Checkpoints saved:

-   model parameters,
-   optimizer state,
-   global training progress,
-   episode rewards,
-   configuration.

This allowed training to resume without starting completely from zero.

------------------------------------------------------------------------

# Part XXVIII --- Limitations

## 62. Limitations of the Study

Important limitations include:

1.  Only a small number of random seeds were explored.
2.  Evaluation used 10 fixed tracks rather than hundreds of tracks.
3.  Training budgets were constrained by Colab/runtime availability.
4.  Scratch and SB3 were not guaranteed to use every low-level
    implementation detail identically.
5.  The Scratch policy showed high evaluation variance.
6.  A single high best episode should not be interpreted as superior
    overall performance.
7.  Hyperparameter optimization was exploratory rather than a large
    systematic sweep.

Acknowledging these limitations makes the project more credible.

------------------------------------------------------------------------

# Part XXIX --- Potential Future Improvements

## 63. Future Work

Possible improvements include:

-   evaluate 3--5 independent seeds systematically,
-   train for a larger interaction budget,
-   use validation-based automatic checkpoint selection,
-   compare CNN architectures,
-   investigate safer bounded-action distributions carefully,
-   experiment with frame skip/action repeat,
-   use observation normalization where appropriate,
-   perform structured hyperparameter sweeps,
-   add TensorBoard/W&B experiment tracking,
-   evaluate on a much larger fixed test set,
-   test recurrent policies,
-   test curriculum or carefully validated reward shaping.

------------------------------------------------------------------------

# Part XXX --- How to Explain This Project in an Interview

## 64. 30-Second Version

> I implemented PPO from scratch in PyTorch for Gymnasium CarRacing-v3
> and compared it with Stable-Baselines3 PPO. My implementation included
> image preprocessing and four-frame stacking, a CNN Gaussian
> actor-critic, GAE, PPO clipping, entropy and value losses, 16 parallel
> environments, learning-rate scheduling, checkpointing and fixed-seed
> evaluation. The selected Scratch checkpoint achieved a 156.5 mean
> reward across 10 fixed tracks with a best episode of 584.4, while SB3
> achieved a higher 220.5 mean. A key finding was that RL performance
> was non-monotonic, so evaluating intermediate checkpoints was
> essential.

------------------------------------------------------------------------

## 65. 60--90 Second Version

> The goal was to understand PPO deeply rather than only call a library.
> CarRacing provides image observations and continuous steering, gas and
> brake controls, so I first preprocess each frame to an 84-by-84
> grayscale image, normalize it, and stack four frames for temporal
> context. A CNN extracts features shared by an actor and critic. The
> actor parameterizes a Gaussian continuous-action policy, while the
> critic predicts state value. I collect 128-step rollouts across 16
> parallel environments, compute GAE advantages with gamma 0.99 and
> lambda 0.95, normalize the advantages, freeze a copy of the old
> policy, and optimize the clipped PPO probability-ratio objective for
> 10 epochs using minibatches. I also include value loss and entropy
> regularization.
>
> One of the interesting engineering lessons was that training wasn't
> monotonic. The Scratch model's training reward improved above 300, but
> later checkpoints generalized worse. I therefore evaluated
> intermediate checkpoints on the same 10 fixed tracks and selected the
> 200k checkpoint, which achieved a 156.5 mean and 584.4 best episode.
> The SB3 baseline achieved a stronger 220.5 mean, showing the
> consistency advantage of a mature implementation. The project taught
> me both the PPO algorithm itself and practical RL debugging around
> seeds, checkpointing, KL spikes, action diagnostics, and evaluation.

------------------------------------------------------------------------

# Part XXXI --- Resume/Portfolio Knowledge Base

## 66. Technologies and Concepts Demonstrated

### Programming / Frameworks

-   Python
-   PyTorch
-   Gymnasium
-   Stable-Baselines3
-   NumPy
-   Matplotlib
-   Google Colab
-   Git/GitHub

### Machine Learning

-   Deep Reinforcement Learning
-   Proximal Policy Optimization
-   Actor-Critic architectures
-   Policy gradients
-   Generalized Advantage Estimation
-   CNN representation learning
-   continuous action spaces
-   stochastic policies
-   entropy regularization
-   value-function approximation
-   parallel environment sampling
-   model checkpointing
-   deterministic evaluation
-   experiment analysis

### Engineering

-   adapting older RL designs to modern APIs,
-   debugging parallel training,
-   reproducible evaluation,
-   checkpoint/resume workflows,
-   seed sensitivity analysis,
-   training diagnostics,
-   qualitative video validation,
-   baseline comparison.

------------------------------------------------------------------------

# Part XXXII --- Resume Bullet Source Material

## 67. Strong Resume Facts

The following are factual project achievements that can be converted
into resume bullets:

-   Implemented **Proximal Policy Optimization from scratch in PyTorch**
    for continuous visual control in Gymnasium `CarRacing-v3`.
-   Built a custom **CNN Gaussian actor-critic** processing four stacked
    `84×84` grayscale frames.
-   Implemented **GAE, PPO clipped objective, value-function learning,
    entropy regularization, frozen old-policy updates, minibatch
    training, and learning-rate scheduling** manually.
-   Parallelized rollout collection across **16 environments**.
-   Developed checkpoint/resume infrastructure for long-running Colab RL
    experiments.
-   Evaluated intermediate checkpoints on **10 fixed tracks** and
    demonstrated that model quality was non-monotonic with training.
-   Selected a \~200k-step Scratch PPO checkpoint achieving **156.5 mean
    reward**, **584.4 best reward**, and **-2.7 worst reward** over the
    fixed evaluation set.
-   Trained and evaluated a Stable-Baselines3 PPO baseline achieving
    **220.5 mean reward** and **547.5 best reward**.
-   Diagnosed RL failure modes using **KL divergence, clipping fraction,
    entropy, action statistics, episode rewards, and gameplay videos**.
-   Investigated multiple policy/training variants and used empirical
    evaluation to reject unsuccessful approaches rather than relying
    only on theoretical assumptions.

------------------------------------------------------------------------

## 68. Example Resume Bullets

These are examples; they can be shortened depending on available resume
space.

**Option A --- technical**

> Implemented PPO from scratch in PyTorch for Gymnasium CarRacing-v3,
> building a CNN Gaussian actor-critic with GAE, clipped policy updates,
> entropy/value losses, and 16 parallel environments; achieved a 156.5
> mean and 584.4 best reward across 10 fixed evaluation tracks.

**Option B --- comparison focused**

> Built and benchmarked a custom PyTorch PPO agent against
> Stable-Baselines3 on visual continuous control; custom PPO reached a
> 584.4 best episode while SB3 achieved a 220.5 mean, with fixed-seed
> evaluation and checkpoint selection used to analyze generalization.

**Option C --- engineering focused**

> Developed an end-to-end deep-RL training pipeline with parallel
> Gymnasium environments, checkpoint/resume support, CNN frame-stack
> preprocessing, PPO/GAE optimization, training diagnostics, and
> gameplay evaluation for CarRacing-v3.

**Option D --- experimentation focused**

> Diagnosed PPO instability using KL divergence, clipping fraction,
> entropy and action-distribution metrics; evaluated multiple policy
> variants and intermediate checkpoints, identifying a \~200k-step model
> that outperformed later checkpoints by over 2× in mean evaluation
> reward.

Do not put all four on one resume. Choose one or two depending on the
role.

------------------------------------------------------------------------

# Part XXXIII --- Questions an Interviewer May Ask About Your Resume Bullet

## 69. Be Ready to Answer These

If this project appears on your resume, expect questions such as:

1.  What exactly did you implement yourself versus use from SB3?
2.  Explain PPO clipping mathematically and intuitively.
3.  What is the probability ratio?
4.  Why do you need an old policy?
5.  What is GAE?
6.  Why use gamma 0.99 and lambda 0.95?
7.  What does the critic learn?
8.  Why use entropy regularization?
9.  Why use Gaussian actions?
10. Why stack four frames?
11. Why grayscale/crop the observations?
12. Why use 16 environments?
13. Why is PPO considered on-policy?
14. What does KL divergence tell you?
15. What does clip fraction tell you?
16. Why did performance deteriorate after improving?
17. Why did you select the 200k checkpoint?
18. Why did different seeds behave differently?
19. Why did SB3 outperform Scratch on average?
20. How would you improve the experiment with more compute?
21. How did you ensure a fair evaluation?
22. Why is best episode reward not enough?
23. How did you debug the circling/off-track policies?
24. What failed experiments did you try, and what did you learn?
25. How would you productionize or reproduce the training pipeline?

Everything required to answer these questions is covered in this
document.

------------------------------------------------------------------------

# Part XXXIV --- Final Project Summary

## 70. Final Summary

This project progressed from basic Scratch PPO experiments to a complete
modern visual continuous-control PPO implementation.

The final Scratch pipeline consists of:

``` text
CarRacing-v3 RGB observation
        ↓
crop to 84×84
        ↓
grayscale
        ↓
normalize [-1,1]
        ↓
4-frame stack
        ↓
CNN feature extractor
        ↓
Gaussian Actor + Value Critic
        ↓
16 parallel environments
        ↓
128-step rollouts
        ↓
GAE
        ↓
advantage normalization
        ↓
frozen old policy
        ↓
PPO clipped objective
        +
value loss
        +
entropy bonus
        ↓
10 epochs of minibatch optimization
        ↓
learning-rate decay
        ↓
checkpointing
        ↓
fixed 10-track evaluation
        ↓
best-checkpoint selection
```

The strongest Scratch checkpoint was selected at approximately **200k
environment interactions** and achieved:

-   **156.5 mean reward**
-   **165.3 standard deviation**
-   **584.4 best episode**
-   **-2.7 worst episode**

The SB3 baseline achieved:

-   **220.5 mean reward**
-   **173.6 standard deviation**
-   **547.5 best episode**
-   **44.4 worst episode**

Therefore the main technical conclusion is:

> A manually implemented PPO agent successfully learned meaningful
> visual continuous-control behavior in CarRacing-v3 and was competitive
> on several tracks, but the mature Stable-Baselines3 implementation
> produced stronger average consistency. The project also demonstrated
> that deep-RL training is stochastic and non-monotonic, making
> diagnostics, seed awareness, checkpointing, fixed evaluation, and
> model selection as important as implementing the PPO equations
> themselves.

For interview preparation, the most important topics to master from this
project are:

**PPO clipping → old/new policy ratio → actor-critic → GAE → CNN/frame
stacking → Gaussian continuous actions → entropy → parallel rollouts →
KL/clip diagnostics → checkpoint selection → fixed-seed evaluation →
seed sensitivity.**

If you can explain those components and connect each one to a concrete
design decision or observed failure in this project, you can defend the
project strongly in an ML/AI/software interview.
