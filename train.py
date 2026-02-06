import os
import argparse
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt

# These imports assume your other files are in a 'src' folder
from src.a2c_agent import ActorCritic, train_step
from src.subprocess_env import SubprocessEnvWrapper

def train_agent_with_time_limit(agent, optimizer, environment, time_limit_hours, save_path):
    time_budget_seconds = time_limit_hours * 3600
    start_time = time.time()
    history = {"episode_reward": [], "episode_duration_sec": [], "episode_length_steps": []}
    print("="*60); print(f"Starting training on '{environment.mode}' for {time_limit_hours} hours."); print("="*60)
    episodes_completed = 0
    try:
        for episode in range(1, 100001):
            episode_start_time = time.time()
            initial_state, _ = environment.reset()
            initial_state = tf.constant(initial_state, dtype=tf.float32)
            rewards_tensor = train_step(initial_state, agent, optimizer, 0.99, environment.T, environment)
            episode_reward = tf.reduce_sum(rewards_tensor).numpy()
            episode_length = tf.shape(rewards_tensor)[0].numpy()
            episode_duration = time.time() - episode_start_time
            history["episode_reward"].append(episode_reward); history["episode_duration_sec"].append(episode_duration); history["episode_length_steps"].append(episode_length)
            episodes_completed += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_budget_seconds:
                print("\n" + "*"*60); print("TRAINING TIME BUDGET REACHED. Stopping."); print("*"*60); break
            print(f"  - Env: {environment.mode.upper()} | Episode: {episodes_completed} | Reward: {episode_reward:.2f} | Elapsed: {elapsed_time/60:.2f} min", end='\r')
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("\n" + "="*60); print("Training Finished."); print(f"Total Elapsed Time: {(time.time() - start_time) / 60:.2f} minutes."); print(f"Total Episodes Completed: {episodes_completed}")
        agent.save_weights(save_path); print(f"Agent weights saved to: {save_path}"); print("="*60)
        environment.close()
        return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an A2C agent on either the SWMM or Surrogate environment.")
    parser.add_argument('--mode', type=str, required=True, choices=['swmm', 'surrogate'], help="The environment mode to train on ('swmm' or 'surrogate').")
    parser.add_argument('--time_budget', type=float, default=5.0, help="The training time budget in hours.")
    args = parser.parse_args()

    # Relative paths to your data and model files
    SWMM_INP_PATH = 'swmm_model/conner_creek.inp'
    SAVED_MODELS_DIR = 'saved_models'
    DATA_DIR = 'data'
    SURROGATE_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'surrogate_model.h5')
    SCALER_PATH = os.path.join(SAVED_MODELS_DIR, 'min_max_scaler.gz')
    RAINFALL_CSV_PATH = os.path.join(DATA_DIR, 'historical_rainfall.csv')

    TRAINING_START_TIME = datetime(2013, 12, 1, 0, 0, 0)
    TRAINING_END_TIME = datetime(2013, 12, 31, 0, 0, 0)

    if args.mode == 'swmm':
        print("\n<<<<<<<<<< CONFIGURING SWMM AGENT TRAINING >>>>>>>>>>")
        env_config = {
            'mode': 'swmm', 'inp_file': SWMM_INP_PATH, 'scaler_path': SCALER_PATH,
            'start_time': TRAINING_START_TIME, 'end_time': TRAINING_END_TIME
        }
    else: # mode == 'surrogate'
        print("\n<<<<<<<<<< CONFIGURING SURROGATE AGENT TRAINING >>>>>>>>>>")
        env_config = {
            'mode': 'surrogate', 'surrogate_model_path': SURROGATE_MODEL_PATH, 'scaler_path': SCALER_PATH,
            'rainfall_csv_path': RAINFALL_CSV_PATH, 'start_time': TRAINING_START_TIME, 'end_time': TRAINING_END_TIME
        }

    output_dir = f"training_run_{args.mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    env = SubprocessEnvWrapper(env_config)

    agent = ActorCritic(num_actions_per_pond=3, num_hidden_units=128)
    optimizer = Adam(learning_rate=0.001)

    dummy_obs = tf.zeros((1, env.observation_space.shape[0])); agent(dummy_obs); optimizer.build(agent.trainable_variables)

    save_path = os.path.join(output_dir, f'agent_{args.mode}.weights.h5')
    history = train_agent_with_time_limit(
        agent=agent, optimizer=optimizer, environment=env,
        time_limit_hours=args.time_budget, save_path=save_path
    )
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, f"history_{args.mode}.csv"), index=False)
    
    print(f"\nTraining complete. History data saved to {os.path.join(output_dir, f'history_{args.mode}.csv')}")
    print(f"\n--- {args.mode.upper()} TRAINING SUMMARY ---"); print(history_df.describe())
    
    plt.figure(figsize=(12, 6))
    rolling_avg_window = 50 if args.mode == 'surrogate' else 5
    plt.plot(history_df['episode_reward'], label='Per-Episode Reward', alpha=0.5)
    plt.plot(history_df['episode_reward'].rolling(window=rolling_avg_window).mean(), label=f'{rolling_avg_window}-Episode Rolling Avg', color='red')
    plt.title(f'{args.mode.upper()} Agent Learning Curve'); plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.legend(); plt.grid(True)
    plot_save_path = os.path.join(output_dir, f'learning_curve_{args.mode}.png')
    plt.savefig(plot_save_path)
    print(f"Learning curve plot saved to: {plot_save_path}")
    plt.show()
