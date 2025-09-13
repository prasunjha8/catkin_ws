#!/usr/bin/env python3

import rospy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from bike_env import BikeEnv
import numpy as np
import time

def test_trained_model():
    """Test the existing trained model"""
    rospy.loginfo("Testing trained model...")
    
    try:
        # Initialize environment
        env = BikeEnv()
        env = DummyVecEnv([lambda: env])
        
        # Load the trained model
        model_path = "/tmp/bike_rl/sac_bike_model"
        rospy.loginfo(f"Loading model from: {model_path}")
        
        model = SAC.load(model_path, env=env)
        rospy.loginfo("Model loaded successfully!")
        
        # Run test episodes
        n_episodes = 3
        for episode in range(n_episodes):
            rospy.loginfo(f"Starting episode {episode + 1}/{n_episodes}")
            
            obs = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 500:  # Limit steps per episode
                # Predict action using trained model
                action, _ = model.predict(obs, deterministic=True)
                
                # Take step
                obs, reward, done, info = env.step(action)
                total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                steps += 1
                
                # Log progress every 50 steps
                if steps % 50 == 0:
                    rospy.loginfo(f"Episode {episode + 1}, Step {steps}: Reward={total_reward:.2f}")
                
                time.sleep(0.01)  # Small delay for visualization
            
            rospy.loginfo(f"Episode {episode + 1} finished: Steps={steps}, Total Reward={total_reward:.2f}, Done={done}")
            
            if episode < n_episodes - 1:  # Don't sleep after last episode
                rospy.loginfo("Waiting 3 seconds before next episode...")
                time.sleep(3)
        
        rospy.loginfo("Testing completed!")
        
    except Exception as e:
        rospy.logerr(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()

def test_random_actions():
    """Test with random actions for comparison"""
    rospy.loginfo("Testing with random actions...")
    
    try:
        env = BikeEnv()
        
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(100):
            action = np.random.uniform(-1, 1, 2)  # Random action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 20 == 0:
                rospy.loginfo(f"Random test step {steps}: Reward={total_reward:.2f}")
            
            if done:
                rospy.loginfo("Episode ended early")
                break
            
            time.sleep(0.05)
        
        rospy.loginfo(f"Random test finished: Steps={steps}, Total Reward={total_reward:.2f}")
        
    except Exception as e:
        rospy.logerr(f"Random test failed: {e}")

if __name__ == '__main__':
    try:
        rospy.loginfo("Choose test mode:")
        rospy.loginfo("1. Test trained model")
        rospy.loginfo("2. Test random actions")
        rospy.loginfo("3. Both")
        
        # For automatic testing, let's do both
        test_random_actions()
        rospy.loginfo("="*50)
        test_trained_model()
        
    except rospy.ROSInterruptException:
        pass