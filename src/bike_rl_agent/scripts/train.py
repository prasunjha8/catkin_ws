#!/usr/bin/env python3

import rospy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from bike_env import MotorcycleEnv
import os
import torch as th
import numpy as np

def main():
    """
    Train SAC agent for realistic motorcycle control
    """
    try:
        rospy.loginfo("Starting realistic motorcycle training...")
        
        # Create motorcycle environment
        env = MotorcycleEnv()
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        # Policy network optimized for motorcycle dynamics
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=[512, 256, 128],  # Larger network for complex dynamics
            n_critics=2,  # SAC uses twin critics
        )
        
        log_dir = "/tmp/motorcycle_rl/"
        os.makedirs(log_dir, exist_ok=True)
        model_path = os.path.join(log_dir, "sac_motorcycle_model")

        # Training callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=25000,
            save_path=log_dir,
            name_prefix="motorcycle_model"
        )
        
        eval_callback = EvalCallback(
            env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
            n_eval_episodes=3
        )
        
        callbacks = [checkpoint_callback, eval_callback]
        
        # Load existing model or create new one
        if os.path.exists(model_path + ".zip"):
            rospy.loginfo("Loading existing motorcycle model...")
            model = SAC.load(model_path, env=env)
            rospy.loginfo("Motorcycle model loaded successfully!")
        else:
            rospy.loginfo("Creating new motorcycle model...")
            
            # Action noise for exploration
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), 
                sigma=0.15 * np.ones(n_actions)  # Moderate exploration noise
            )
            
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=300000,  # Large buffer for diverse experiences
                batch_size=512,     # Larger batches for stability
                gamma=0.99,         # Standard discount factor
                tau=0.005,          # Soft update rate
                ent_coef='auto',    # Automatic entropy tuning
                learning_starts=15000,  # Collect experience before training
                policy_kwargs=policy_kwargs,
                action_noise=action_noise,
                tensorboard_log=None,
                device='cpu'
            )
            rospy.loginfo("Motorcycle model created successfully!")

        # Start training
        rospy.loginfo("Starting motorcycle training...")
        rospy.loginfo("The motorcycle will learn to:")
        rospy.loginfo("1. Maintain balance through speed")
        rospy.loginfo("2. Use counter-steering for turns") 
        rospy.loginfo("3. Navigate to the target")
        
        try:
            model.learn(
                total_timesteps=200000,  # Extended training for complex dynamics
                callback=callbacks,
                log_interval=5,
                progress_bar=True
            )
            rospy.loginfo("Training completed successfully!")
            
        except KeyboardInterrupt:
            rospy.logwarn("Training interrupted by user.")
        except Exception as e:
            rospy.logerr(f"Training failed: {e}")
            raise

        # Save final model
        rospy.loginfo("Saving motorcycle model...")
        model.save(model_path)
        rospy.loginfo(f"Model saved to {model_path}")

        # Test the trained motorcycle
        rospy.loginfo("Testing trained motorcycle...")
        test_motorcycle_model(model, env, episodes=5)

    except rospy.ROSInterruptException:
        rospy.logwarn("ROS node interrupted.")
    except Exception as e:
        rospy.logerr(f"Training script failed: {e}")
        raise
    finally:
        if 'env' in locals():
            try:
                env.close()
                rospy.loginfo("Environment closed.")
            except:
                pass

def test_motorcycle_model(model, env, episodes=5):
    """Test the trained motorcycle model"""
    rospy.loginfo(f"Running {episodes} motorcycle test episodes...")
    
    successful_episodes = 0
    total_distance_covered = 0
    
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False
        max_speed = 0
        initial_distance = 20.0  # Starting distance to goal
        
        rospy.loginfo(f"Starting motorcycle test episode {episode + 1}")
        
        while not done and steps < 1000:
            # Use deterministic policy for testing
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            
            # Track statistics
            if len(info) > 0 and isinstance(info[0], dict):
                current_speed = info[0].get('speed', 0)
                max_speed = max(max_speed, current_speed)
            
            steps += 1
            
            # Log progress every 100 steps
            if steps % 100 == 0:
                current_distance = obs[0][16] if hasattr(obs[0], '__len__') else obs[16]
                rospy.loginfo(f"Episode {episode + 1}, Step {steps}: "
                             f"Reward={total_reward:.1f}, Distance={current_distance:.1f}m")
        
        # Episode statistics
        if len(obs) > 0:
            final_distance = obs[0][16] if hasattr(obs[0], '__len__') else obs[16]
            distance_covered = initial_distance - final_distance
            total_distance_covered += distance_covered
            
            if final_distance < 5.0:  # Consider success if within 5m
                successful_episodes += 1
            
            rospy.loginfo(f"Episode {episode + 1} complete:")
            rospy.loginfo(f"  Steps: {steps}")
            rospy.loginfo(f"  Total Reward: {total_reward:.1f}")
            rospy.loginfo(f"  Max Speed: {max_speed:.1f} m/s")
            rospy.loginfo(f"  Distance Covered: {distance_covered:.1f} m")
            rospy.loginfo(f"  Final Distance to Goal: {final_distance:.1f} m")
    
    # Overall statistics
    success_rate = (successful_episodes / episodes) * 100
    avg_distance = total_distance_covered / episodes
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("MOTORCYCLE TEST RESULTS:")
    rospy.loginfo(f"Success Rate: {success_rate:.1f}%")
    rospy.loginfo(f"Average Distance Covered: {avg_distance:.1f} m")
    rospy.loginfo(f"Successful Episodes: {successful_episodes}/{episodes}")
    rospy.loginfo("=" * 50)

if __name__ == '__main__':
    main()