#!/usr/bin/env python3

import rospy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from bike_env import BikeEnv
import os
import torch as th
import numpy as np

def main():
    """
    Main function to initialize the environment, agent, and start training.
    """
    try:
        # Gym Environment
        rospy.loginfo("Initializing environment...")
        env = BikeEnv()
        env = Monitor(env)  # monitoring training progress
        
        # Wrap in DummyVecEnv for compatibility
        env = DummyVecEnv([lambda: env])

        # 2. Define the RL Agent (SAC)
        
        # Custom Policy Network Architecture
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=[256, 256]  # Two hidden layers with 256 neurons each
        )
        
        # Directory for saving models and logs
        log_dir = "/tmp/bike_rl/"
        os.makedirs(log_dir, exist_ok=True)
        model_path = os.path.join(log_dir, "sac_bike_model")

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,  # Save every 10k steps
            save_path=log_dir,
            name_prefix="bike_rl_model"
        )
        
        # Evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Combine callbacks
        callbacks = [checkpoint_callback, eval_callback]
        
        # Instantiate the SAC agent
        if os.path.exists(model_path + ".zip"):
            rospy.loginfo("Loading existing model...")
            model = SAC.load(model_path, env=env)
            rospy.loginfo("Model loaded successfully!")
        else:
            rospy.loginfo("Creating a new model...")
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=100000,  # Reduced for memory efficiency
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                ent_coef='auto',
                learning_starts=5000,  # Reduced for faster initial learning
                policy_kwargs=policy_kwargs,
                tensorboard_log=None  # Disable tensorboard for now
            )
            rospy.loginfo("Model created successfully!")

        # 3. Start the Training Process
        rospy.loginfo("Starting training...")
        
        # Train for fewer steps initially to test
        total_timesteps = 50000  # Reduced for initial testing
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=10,
                progress_bar=True
            )
            rospy.loginfo("Training completed successfully!")
            
        except KeyboardInterrupt:
            rospy.logwarn("Training interrupted by user.")
        except Exception as e:
            rospy.logerr(f"Training failed with error: {e}")
            raise

        # 4. Save the final trained model
        rospy.loginfo("Saving final model...")
        model.save(model_path)
        rospy.loginfo(f"Model saved to {model_path}")

        # 5. Test the trained model
        rospy.loginfo("Testing the trained model...")
        test_trained_model(model, env, episodes=3)

    except rospy.ROSInterruptException:
        rospy.logwarn("ROS node interrupted.")
    except Exception as e:
        rospy.logerr(f"Training script failed: {e}")
        raise
    finally:
        if 'env' in locals():
            try:
                env.close()
                rospy.loginfo("Environment closed successfully.")
            except:
                pass

def test_trained_model(model, env, episodes=5):
    """
    Test the trained model for a few episodes
    """
    rospy.loginfo(f"Running {episodes} test episodes...")
    
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:  # Limit steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1
            
        rospy.loginfo(f"Episode {episode + 1}: Steps={steps}, Total Reward={total_reward:.2f}")

if __name__ == '__main__':
    main()