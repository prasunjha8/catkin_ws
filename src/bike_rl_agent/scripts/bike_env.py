#!/usr/bin/env python3

import gym
from gym import spaces
import numpy as np
import rospy
import rosservice
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import Empty
import tf.transformations as tft
import math
import time
import threading

class BikeEnv(gym.Env):
    """Custom OpenAI Gym environment for the self-balancing bike in Gazebo."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BikeEnv, self).__init__()
        
        # Initialize ROS node if not already initialized
        if not rospy.get_node_uri():
            rospy.init_node('bike_env', anonymous=True)
        
        rospy.loginfo("Initializing BikeEnv...")

        # Action and Observation Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        # ROS Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Thread safety
        self._data_lock = threading.Lock()
        
        # Initialize ALL state variables BEFORE setting up subscribers
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        
        self.bike_pose = None
        self.bike_twist = None
        self.bike_model_index = -1  # This MUST be initialized before subscribers
        
        self.steering_angle = 0.0
        self.rear_wheel_vel = 0.0

        # Target and episode parameters
        self.target_pos = np.array([10.0, 0.0])
        self.prev_distance_to_goal = 10.0
        self.step_count = 0
        self.max_steps = 1000

        # Setup with timeouts
        self._setup_with_retry()
        
        rospy.loginfo("BikeEnv initialized successfully.")

    def _setup_with_retry(self, max_retries=3):
        """Setup ROS connections with retry logic"""
        for attempt in range(max_retries):
            try:
                rospy.loginfo(f"Setup attempt {attempt + 1}/{max_retries}")
                
                # Check if ROS master is running
                if not self._check_ros_master():
                    rospy.logwarn("ROS master not responding")
                    time.sleep(2)
                    continue
                
                # Setup Gazebo services with timeout
                if not self._setup_gazebo_services_with_timeout():
                    rospy.logwarn("Failed to connect to Gazebo services")
                    time.sleep(2)
                    continue
                
                # Setup subscribers
                self._setup_subscribers()
                
                # Wait for initial data with timeout
                if not self._wait_for_initial_data_with_timeout():
                    rospy.logwarn("No initial sensor data received, but continuing...")
                
                return  # Success
                
            except Exception as e:
                rospy.logwarn(f"Setup attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    rospy.logwarn("All setup attempts failed. Environment may not work correctly.")
                else:
                    time.sleep(2)

    def _check_ros_master(self):
        """Check if ROS master is running"""
        try:
            rospy.get_master().getSystemState()
            return True
        except:
            return False

    def _setup_gazebo_services_with_timeout(self, timeout=10):
        """Setup Gazebo services with timeout"""
        services = [
            '/gazebo/reset_simulation',
            '/gazebo/unpause_physics', 
            '/gazebo/pause_physics'
        ]
        
        for service_name in services:
            try:
                rospy.loginfo(f"Waiting for service: {service_name}")
                rospy.wait_for_service(service_name, timeout=timeout)
            except rospy.ROSException:
                rospy.logwarn(f"Service {service_name} not available after {timeout}s")
                return False
        
        try:
            self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
            self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
            rospy.loginfo("Gazebo services connected successfully.")
            return True
        except Exception as e:
            rospy.logwarn(f"Failed to create service proxies: {e}")
            return False

    def _setup_subscribers(self):
        """Setup ROS subscribers"""
        rospy.Subscriber('/imu', Imu, self._imu_callback, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self._model_states_callback, queue_size=1)

    def _wait_for_initial_data_with_timeout(self, timeout=15):
        """Wait for initial sensor data with timeout"""
        rospy.loginfo("Waiting for initial sensor data...")
        start_time = time.time()
        
        # First, check if topics exist
        topics = rospy.get_published_topics()
        topic_names = [topic[0] for topic in topics]
        
        if '/imu' not in topic_names:
            rospy.logwarn("IMU topic not found in published topics")
        if '/gazebo/model_states' not in topic_names:
            rospy.logwarn("Model states topic not found in published topics")
        
        # Wait for data
        while (time.time() - start_time) < timeout and not rospy.is_shutdown():
            with self._data_lock:
                if self.bike_pose is not None and self.bike_twist is not None:
                    rospy.loginfo("Initial sensor data received.")
                    return True
            time.sleep(0.1)
        
        rospy.logwarn(f"No initial sensor data after {timeout}s. Available topics:")
        for topic in topic_names[:10]:  # Show first 10 topics
            rospy.logwarn(f"  {topic}")
        return False

    def _imu_callback(self, data):
        """IMU callback with error handling"""
        try:
            with self._data_lock:
                orientation_q = data.orientation
                orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                (self.roll, self.pitch, self.yaw) = tft.euler_from_quaternion(orientation_list)
                
                self.roll_rate = data.angular_velocity.x
                self.pitch_rate = data.angular_velocity.y
                self.yaw_rate = data.angular_velocity.z
        except Exception as e:
            rospy.logwarn_once(f"IMU callback error: {e}")

    def _model_states_callback(self, data):
        """Model states callback with error handling"""
        try:
            with self._data_lock:
                if self.bike_model_index == -1:
                    # Look for bike model
                    possible_names = ['two_wheel_bike', 'bike', 'robot']
                    for name in possible_names:
                        try:
                            self.bike_model_index = data.name.index(name)
                            rospy.loginfo(f"Found bike model: {name} at index {self.bike_model_index}")
                            break
                        except ValueError:
                            continue
                    
                    if self.bike_model_index == -1:
                        rospy.logwarn_once(f"Bike model not found. Available models: {data.name}")
                        return
                
                if (self.bike_model_index < len(data.pose) and 
                    self.bike_model_index < len(data.twist)):
                    self.bike_pose = data.pose[self.bike_model_index]
                    self.bike_twist = data.twist[self.bike_model_index]
        except Exception as e:
            rospy.logwarn_once(f"Model states callback error: {e}")

    def _get_obs(self):
        """Get observation with fallback for missing data"""
        with self._data_lock:
            # Use default values if data not available
            if self.bike_pose is None:
                # Create default pose
                from geometry_msgs.msg import Pose, Point, Quaternion
                self.bike_pose = Pose()
                self.bike_pose.position = Point(0, 0, 0.4)
                self.bike_pose.orientation = Quaternion(0, 0, 0, 1)
            
            if self.bike_twist is None:
                # Create default twist
                from geometry_msgs.msg import Twist, Vector3
                self.bike_twist = Twist()
                self.bike_twist.linear = Vector3(0, 0, 0)
                self.bike_twist.angular = Vector3(0, 0, 0)

            # 1. IMU Data (6)
            imu_data = [self.roll, self.pitch, self.yaw, 
                       self.roll_rate, self.pitch_rate, self.yaw_rate]
            
            # 2. Body Velocities (6)
            body_vel = [
                self.bike_twist.linear.x, self.bike_twist.linear.y, self.bike_twist.linear.z,
                self.bike_twist.angular.x, self.bike_twist.angular.y, self.bike_twist.angular.z
            ]

            # 3. Joint States (2)
            joint_states = [self.steering_angle, self.rear_wheel_vel]

            # 4. Target Information (2)
            current_pos = np.array([self.bike_pose.position.x, self.bike_pose.position.y])
            vector_to_goal = self.target_pos - current_pos
            distance_to_goal = np.linalg.norm(vector_to_goal)
            
            if distance_to_goal > 0.001:  # Avoid division by zero
                angle_to_goal = math.atan2(vector_to_goal[1], vector_to_goal[0])
                bearing = angle_to_goal - self.yaw
                # Normalize bearing
                while bearing > math.pi:
                    bearing -= 2 * math.pi
                while bearing < -math.pi:
                    bearing += 2 * math.pi
            else:
                bearing = 0.0
                
            target_info = [distance_to_goal, bearing]

            # Combine all observations
            obs = np.array(imu_data + body_vel + joint_states + target_info, dtype=np.float32)
            obs = np.clip(obs, -100, 100)  # Prevent extreme values
            
            return obs

    def step(self, action):
        """Environment step with error handling"""
        try:
            # Unpause physics
            try:
                self.unpause_proxy()
            except:
                rospy.logwarn_once("Failed to unpause physics")

            # Process action
            self.rear_wheel_vel = float(np.clip(action[0], -1, 1)) * 5.0
            self.steering_angle = float(np.clip(action[1], -1, 1)) * 0.52

            # Send command
            cmd = Twist()
            cmd.linear.x = self.rear_wheel_vel
            cmd.angular.z = self.steering_angle
            self.cmd_vel_pub.publish(cmd)
            
            # Wait for simulation
            time.sleep(0.1)

            # Pause physics
            try:
                self.pause_proxy()
            except:
                rospy.logwarn_once("Failed to pause physics")

            self.step_count += 1
            
            # Get observation and compute reward
            obs = self._get_obs()
            reward = self._compute_reward(obs)
            done = False

            # Check termination conditions
            if abs(self.roll) > 1.0:  # Fallen over
                done = True
                reward = -100.0
            elif len(obs) >= 15 and obs[14] < 1.0:  # Reached goal
                done = True
                reward += 100.0
            elif self.step_count >= self.max_steps:
                done = True

            # Small reward for staying alive
            if not done:
                reward += 1.0

            return obs, float(reward), done, {}

        except Exception as e:
            rospy.logwarn(f"Step error: {e}")
            # Return safe defaults
            obs = self._get_obs()
            return obs, -1.0, True, {}

    def _compute_reward(self, obs):
        """Simple reward function"""
        try:
            if len(obs) < 15:
                return 0.0
                
            # Balance reward
            balance_reward = math.exp(-5.0 * abs(self.roll))
            
            # Forward motion
            forward_vel = obs[6] if len(obs) > 6 else 0.0
            forward_reward = max(0, forward_vel) * 0.5
            
            # Progress toward goal
            current_distance = obs[14]
            progress_reward = (self.prev_distance_to_goal - current_distance) * 10.0
            self.prev_distance_to_goal = current_distance
            
            return balance_reward + forward_reward + progress_reward
            
        except Exception as e:
            rospy.logwarn_once(f"Reward computation error: {e}")
            return 0.0

    def reset(self):
        """Reset environment with error handling"""
        try:
            rospy.loginfo("Resetting environment...")
            
            # Reset simulation
            try:
                self.reset_simulation_proxy()
                time.sleep(1.0)  # Wait for reset
            except:
                rospy.logwarn("Failed to reset simulation")
            
            # Reset internal state
            self.step_count = 0
            self.steering_angle = 0.0
            self.rear_wheel_vel = 0.0
            
            # Stop the robot
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            
            # Unpause briefly to get fresh data
            try:
                self.unpause_proxy()
                time.sleep(0.2)
                self.pause_proxy()
            except:
                pass
            
            # Get fresh observation
            obs = self._get_obs()
            self.prev_distance_to_goal = obs[14] if len(obs) >= 15 else 10.0
            
            return obs
            
        except Exception as e:
            rospy.logwarn(f"Reset error: {e}")
            return self._get_obs()

    def close(self):
        """Clean shutdown"""
        rospy.loginfo("Closing BikeEnv...")
        try:
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
        except:
            pass

if __name__ == '__main__':
    # Test the environment
    try:
        env = BikeEnv()
        obs = env.reset()
        rospy.loginfo("Environment test successful!")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            rospy.loginfo(f"Step {i}: Reward={reward:.2f}, Done={done}")
            if done:
                obs = env.reset()
                
    except rospy.ROSInterruptException:
        pass