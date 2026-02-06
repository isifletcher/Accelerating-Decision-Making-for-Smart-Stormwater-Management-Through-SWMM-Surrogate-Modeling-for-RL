import tensorflow as tf
from tensorflow.keras import layers, losses
import numpy as np

class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions_per_pond, num_hidden_units):
        super().__init__()
        self.common1 = layers.Dense(num_hidden_units, activation="relu")
        self.common2 = layers.Dense(num_hidden_units, activation="relu")
        self.actor_heads = [layers.Dense(num_actions_per_pond) for _ in range(4)]
        self.critic = layers.Dense(1)
    def call(self, inputs: tf.Tensor) -> tuple[list[tf.Tensor], tf.Tensor]:
        x = self.common1(inputs); x = self.common2(x)
        return [head(x) for head in self.actor_heads], self.critic(x)

def run_episode(initial_state: tf.Tensor, agent: tf.keras.Model, max_steps: int, env):
    def env_step_wrapper(action_1, action_2, action_3, action_4):
        action_np = np.array([a.numpy() for a in [action_1, action_2, action_3, action_4]])
        obs, reward, terminated, truncated, _ = env.step(action_np)
        return np.array(obs, dtype=np.float32), np.array(reward, dtype=np.float32), np.array(terminated, dtype=np.bool_), np.array(truncated, dtype=np.bool_)
    
    action_log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    state = initial_state
    
    for t in tf.range(max_steps):
        state_tensor = tf.expand_dims(state, 0)
        action_logits_list, value = agent(state_tensor)
        actions, log_probs = [], []
        for logits in action_logits_list:
            action = tf.random.categorical(logits, 1)[0, 0]
            actions.append(action)
            log_probs.append(tf.math.log(tf.nn.softmax(logits)[0, action]))
        action_log_probs = action_log_probs.write(t, tf.reduce_sum(log_probs))
        values = values.write(t, tf.squeeze(value))
        obs, reward, terminated, truncated = tf.py_function(
            func=env_step_wrapper, inp=actions, Tout=[tf.float32, tf.float32, tf.bool, tf.bool])
        state = tf.reshape(obs, state.shape); reward = tf.reshape(reward, [])
        terminated = tf.reshape(terminated, []); truncated = tf.reshape(truncated, [])
        rewards = rewards.write(t, reward)
        if terminated or truncated: break
    return action_log_probs.stack(), values.stack(), rewards.stack()

def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
    n = tf.shape(rewards)[0]; returns = tf.TensorArray(dtype=tf.float32, size=n)
    rewards = tf.cast(rewards[::-1], dtype=tf.float32); discounted_sum = 0.0
    for i in tf.range(n):
        discounted_sum = rewards[i] + gamma * discounted_sum
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]
    if standardize: returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)
    return returns

huber_loss = losses.Huber(reduction="sum")
def compute_loss(action_log_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    advantage = returns - values
    actor_loss = -tf.math.reduce_sum(action_log_probs * tf.stop_gradient(advantage))
    critic_loss = huber_loss(values, returns)
    return actor_loss + critic_loss

@tf.function
def train_step(initial_state: tf.Tensor, agent: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, gamma: float, max_steps_per_episode: int, env) -> tf.Tensor:
    with tf.GradientTape() as tape:
        action_log_probs, values, rewards = run_episode(initial_state, agent, max_steps_per_episode, env)
        returns = get_expected_return(rewards, gamma)
        loss = compute_loss(action_log_probs, values, returns)
    grads = tape.gradient(loss, agent.trainable_variables)
    optimizer.apply_gradients(zip(grads, agent.trainable_variables))
    return rewards
