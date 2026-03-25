CS182 Study Notes: Imitation Learning and Learning-Based Control

1. The Paradigm Shift: Prediction vs. Control

Transitioning from standard supervised deep learning to learning-based control requires shifting from "learning to predict" to "learning to make decisions." In prediction, we assume data is independent and identically distributed (IID). Formally, the probability of a dataset \mathcal{D} is: p(\mathcal{D}) = \prod_i p(y_i|x_i)p(x_i) In control, the IID assumption is fundamentally broken: your action a_t at time t influences the observation o_{t+1} at time t+1.

Comparison: Prediction vs. Control

Dimension	Prediction Problems	Control Problems
Data Distribution	IID (Static mapping)	Non-IID (Feedback loop)
Supervision	Ground Truth Labels (e.g., "puppy")	Abstract Goals (e.g., "reduce congestion")
Objective	Maximize Average Accuracy	Task Accomplishment / Reward
Input-Output Feedback	None (y_1 does not affect x_2)	High (a_t changes future o_{t+1})

The Mountain Road Analogy: In image classification, if you misidentify a "Leopard" as a "Tiger," the next image in your dataset remains unchanged. In control, imagine driving along a windy mountain road. If you swerve slightly (an "incorrect" action), you are now in a new physical position you wouldn't have otherwise visited. This change in state alters every subsequent input you receive. A single mistake doesn't just result in one wrong label—it changes the entire future distribution of the data.

TA NOTE: Success Metrics Do not be fooled by your training logs. In imitation learning, a low Negative Log-Likelihood (NLL) or high training accuracy does not guarantee success. Our ultimate metric is task reward—whether the car actually stays on the road—not how well we predicted the expert's labels on the training set.


--------------------------------------------------------------------------------


2. Mathematical Formalism: MDPs and POMDPs

To design control systems, we use the framework of Markov Decision Processes (MDPs) to model the configuration of the world.

2.1 Markov Decision Process (MDP)

An MDP is formally defined by (S, A, P, r):

* S: State space (the underlying configuration of the world).
* A: Action space (decisions the agent makes).
* P(s_{t+1}|s_t, a_t): Transition function (probability of the next state given current state and action).
* r(s, a): Reward function (measure of success).

2.2 The Markov Property

A state s_t satisfies the Markov Property if the future is conditionally independent of the past, given the present. In other words, s_t summarizes everything necessary to predict s_{t+1}. If a system is Markovian, the optimal policy \pi(a_t|s_t) only requires the current state, not the history of previous states.

2.3 Partially Observed MDPs (POMDPs)

In most engineering applications, we lack access to the true state s_t (physical reality) and must rely on observations o_t (pixels, sensor readings).

The Cheetah and Gazelle Example:

* State (s_t): The exact muscle tensions, velocities, and 3D coordinates of both animals.
* Observation (o_t): A 2D camera image. If a car drives between the camera and the cheetah, the observation changes (pixels no longer show the predator), but the underlying state s_t remains—the cheetah is still physically there. Because o_t is an incomplete summary of s_t, it does not satisfy the Markov property. Thus, effective policies in POMDPs must depend on the full history of observations: \pi(a_t | o_1, \dots, o_t).


--------------------------------------------------------------------------------


3. Imitation Learning Fundamentals: Behavioral Cloning (BC)

Behavioral Cloning (BC) is the simplest way to apply deep learning to control by "cloning" an expert.

* Formal Objective Function: We treat control as a supervised regression/classification task, maximizing the likelihood of expert actions: \max_{\theta} \sum_{s,a \sim D} \log \pi_{\theta}(a|s)
* The Recipe: Collect human demonstrations, create a dataset of (o_t, a_t) pairs, and train a policy (e.g., a ConvNet).

TA NOTE: Shape Checks and "Vibecoding" Before you hit "Train," perform a manual Shape Check. If you don't know the exact dimensions of your tensors at every layer—especially when dealing with batch sizes and temporal sequences—you are just "vibecoding." Once the dimensions are set, ensure your Softmax implementation is numerically stable (using the LogSumExp trick) to prevent gradients from exploding.


--------------------------------------------------------------------------------


4. The Core Failure of BC: Distributional Shift

Behavioral Cloning often fails because of Distributional Shift: the discrepancy between the training distribution P_{data}(o_t) and the execution distribution P_{\pi_{\theta}}(o_t).

4.1 Compounding Errors

Why does the error grow quadratically (O(T^2))?

1. The model makes a small mistake \epsilon at t=1.
2. The resulting state at t=2 is slightly atypical, making a larger mistake more likely.
3. The total error at time t is roughly the sum of all mistakes made up to that point. Over a trajectory of length T, the accumulated error is \sum_{t=1}^T \epsilon t, which leads to O(T^2) growth.

4.2 The RNN Analogy

This shift is identical to the "training-test discrepancy" in Recurrent Neural Networks (RNNs) using Teacher Forcing. During training, the RNN sees ground-truth tokens; at test time, it sees its own (potentially wrong) previous outputs. While Scheduled Sampling (feeding model predictions back during training) helps RNNs, it is difficult in the physical world because we don't know the world's transition probabilities P(s_{t+1}|s_t, a_t) without actually interacting with it.


--------------------------------------------------------------------------------


5. Practical Enhancements for Behavioral Cloning

5.1 Handling Non-Markovian Behavior

Humans have memory and emotions; a driver’s action might depend on a car that cut them off three minutes ago. To model this, we use an architecture with history, such as an RNN with a convolutional encoder (Many-to-One). This allows the model to process a sequence of frames into a latent memory state before outputting an action.

5.2 Handling Multimodal Behavior (MSE Failure)

If a drone faces a tree, the expert may go left 50% of the time or right 50%. Mean Squared Error (MSE) will force the model to average these, resulting in the drone flying straight into the tree.

Solutions:

* Mixture of Gaussians (MoG): Output means (\mu), variances (\sigma), and weights (w) for multiple peaks.
  * Warning: Training MoGs is "nasty" because the objective involves the log of a sum. Use standard library functions that implement the LogSumExp trick to maintain numerical stability.
* Latent Variable Models: Use CVAEs or Normalizing Flows to capture unobserved "intent" or "feelings."
* Autoregressive Discretization: To avoid the Curse of Dimensionality (where bins grow exponentially with action dimensions), discretize one dimension at a time. Sample the first dimension, then use it as input to predict the second, similar to sampling words in a sentence.

5.3 Data Augmentation: The NVIDIA 3-Camera Hack

Case Study: Recovery Data NVIDIA researchers used a three-camera setup (Left, Center, Right) to mitigate shift. The center camera used human labels. The Left camera image was labeled with a "turn right" command, and the Right camera with a "turn left" command. This synthetic data taught the car how to recover when it veered off-center, providing training coverage for "atypical states" before they led to O(T^2) failure.


--------------------------------------------------------------------------------


6. The DAgger Algorithm: Dataset Aggregation

DAgger is a principled iterative loop designed to force P_{data}(o_t) = P_{\pi_{\theta}}(o_t).

1. Train \pi_{\theta}(a_t|o_t) from initial human demonstrations.
2. Run the policy \pi_{\theta} in the environment to collect new trajectories (where the model likely makes mistakes).
3. Label those new trajectories by asking a human expert: "What action would you have taken in this state?"
4. Aggregate the new data into the total dataset and Retrain.

Trade-offs: While DAgger solves distributional shift, it is extremely expensive as it requires a human expert to be "on call" to label the model's (often nonsensical) trajectories.


--------------------------------------------------------------------------------


7. Summary of Key Notation

Note that the notation varies between Reinforcement Learning (RL) and Classical Control.

Concept	RL Notation	Control Notation
State	s	x
Action	a	u
Observation	o	y
Policy	\pi	\pi or K

Pedagogical Note: Why "u" for control? It originates from the Russian word for "control" (upravleniye), a convention popularized in the 1950s that persists in modern engineering literature.
