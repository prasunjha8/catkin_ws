#!/usr/bin/env python3

import gym
from gym import spaces
import numpy as np
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import Empty
import tf.transformations as tft
import math
import time
import threading

class MotorcycleEnv(gym.Env):
    """Realistic motorcycle environment with proper bike physics"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MotorcycleEnv, self).__init__()
        
        if not rospy.get_node_uri():
            rospy.init_node('motorcycle_env', anonymous=True)
        
        rospy.loginfo("Initializing Realistic MotorcycleEnv...")

        # Action space: [throttle, steering_angle]
        # Throttle: -1 (full brake/reverse) to +1 (full throttle)
        # Steering: -1 (full left) to +1 (full right)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        # Publishers for motorcycle control
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.steering_pub = rospy.Publisher('/steering_controller/command', Float64, queue_size=1)
        
        # Thread safety
        self._data_lock = threading.Lock()
        
        # State variables
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        
        self.bike_pose = None
        self.bike_twist = None
        self.bike_model_index = -1
        
        # Control state
        self.throttle = 0.0
        self.steering_angle = 0.0
        self.current_speed = 0.0
        self.wheel_angular_velocity = 0.0
        
        # Physics parameters
        self.min_stable_speed = 2.0  # m/s - minimum speed for stability
        self.max_speed = 15.0        # m/s - maximum speed
        self.wheelbase = 1.4         # m - distance between wheels
        
        # Episode management
        self.target_pos = np.array([20.0, 0.0])  # Further target for motorcycle
        self.prev_distance_to_goal = 20.0
        self.step_count = 0
        self.max_steps = 3000  # Longer episodes for motorcycle
        self.episode_count = 0
        
        # Motorcycle physics constants
        self.gravity = 9.81
        self.bike_height = 0.7  # Center of mass height
        
        self._setup_ros_connections()
        rospy.loginfo("Realistic MotorcycleEnv initialized successfully.")

    def _setup_ros_connections(self):
        """Setup ROS connections for motorcycle"""
        try:
            # Wait for Gazebo services
            rospy.wait_for_service('/gazebo/reset_simulation', timeout=10)
            rospy.wait_for_service('/gazebo/unpause_physics', timeout=10)
            rospy.wait_for_service('/gazebo/pause_physics', timeout=10)
            
            self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
            self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
            
            # Subscribe to sensors
            rospy.Subscriber('/imu', Imu, self._imu_callback, queue_size=1)
            rospy.Subscriber('/gazebo/model_states', ModelStates, self._model_states_callback, queue_size=1)
            rospy.Subscriber('/joint_states', JointState, self._joint_states_callback, queue_size=1)
            
            rospy.loginfo("Motorcycle ROS connections established.")
            
        except Exception as e:
            rospy.logerr(f"Failed to setup ROS connections: {e}")
            raise

    def _imu_callback(self, data):
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
        try:
            with self._data_lock:
                if self.bike_model_index == -1:
                    possible_names = ['two_wheel_bike', 'motorcycle', 'bike']
                    for name in possible_names:
                        try:
                            self.bike_model_index = data.name.index(name)
                            break
                        except ValueError:
                            continue
                
                if (self.bike_model_index >= 0 and 
                    self.bike_model_index < len(data.pose) and 
                    self.bike_model_index < len(data.twist)):
                    self.bike_pose = data.pose[self.bike_model_index]
                    self.bike_twist = data.twist[self.bike_model_index]
                    self.current_speed = math.sqrt(
                        self.bike_twist.linear.x**2 + 
                        self.bike_twist.linear.y**2
                    )
        except Exception as e:
            rospy.logwarn_once(f"Model states callback error: {e}")

    def _joint_states_callback(self, data):
        """Get joint states for wheel and steering information"""
        try:
            with self._data_lock:
                if 'rear_wheel_joint' in data.name:
                    idx = data.name.index('rear_wheel_joint')
                    self.wheel_angular_velocity = data.velocity[idx]
                
                if 'steering_joint' in data.name:
                    idx = data.name.index('steering_joint')
                    self.steering_angle = data.position[idx]
        except Exception as e:
            rospy.logwarn_once(f"Joint states callback error: {e}")

    def _get_obs(self):
        """Get motorcycle observation with physics-based features"""
        with self._data_lock:
            if self.bike_pose is None:
                from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3
                self.bike_pose = Pose()
                self.bike_pose.position = Point(0, 0, 0.5)
                self.bike_pose.orientation = Quaternion(0, 0, 0, 1)
                
            if self.bike_twist is None:
                self.bike_twist = Twist()
                self.bike_twist.linear = Vector3(0, 0, 0)
                self.bike_twist.angular = Vector3(0, 0, 0)

            # Core motorcycle state (6)
            orientation_data = [self.roll, self.pitch, self.yaw, 
                               self.roll_rate, self.pitch_rate, self.yaw_rate]
            
            # Velocity state (6)
            velocity_data = [
                self.bike_twist.linear.x, self.bike_twist.linear.y, self.bike_twist.linear.z,
                self.bike_twist.angular.x, self.bike_twist.angular.y, self.bike_twist.angular.z
            ]

            # Control state (2)
            control_data = [self.throttle, self.steering_angle]

            # Motorcycle-specific physics (2)
            physics_data = [
                self.current_speed,
                self.wheel_angular_velocity
            ]

            # Target navigation (2) 
            current_pos = np.array([self.bike_pose.position.x, self.bike_pose.position.y])
            vector_to_goal = self.target_pos - current_pos
            distance_to_goal = np.linalg.norm(vector_to_goal)
            
            if distance_to_goal > 0.001:
                angle_to_goal = math.atan2(vector_to_goal[1], vector_to_goal[0])
                bearing = angle_to_goal - self.yaw
                # Normalize bearing
                while bearing > math.pi:
                    bearing -= 2 * math.pi
                while bearing < -math.pi:
                    bearing += 2 * math.pi
            else:
                bearing = 0.0
                
            target_data = [distance_to_goal, bearing]

            obs = np.array(orientation_data + velocity_data + control_data + 
                          physics_data + target_data, dtype=np.float32)
            obs = np.clip(obs, -100, 100)
            
            return obs

    def step(self, action):
        """Motorcycle physics step"""
        try:
            # Clip and process actions
            action = np.clip(action, -1, 1)
            target_throttle = action[0]
            target_steering = action[1] * 0.5  # ±28.6 degrees max steering
            
            # Unpause physics
            try:
                self.unpause_proxy()
            except:
                pass

            # Apply motorcycle control
            self.throttle = target_throttle
            
            # Map throttle to velocity command (motorcycle style)
            if self.throttle > 0:
                # Forward throttle
                target_velocity = self.throttle * self.max_speed
            else:
                # Braking/reverse
                target_velocity = self.throttle * 5.0  # Slower reverse
            
            # Send drive command
            cmd = Twist()
            cmd.linear.x = target_velocity
            cmd.angular.z = 0.0  # Don't use angular.z for steering
            self.cmd_vel_pub.publish(cmd)
            
            # Send steering command
            steering_cmd = Float64()
            steering_cmd.data = target_steering
            self.steering_pub.publish(steering_cmd)
            
            # Simulation timestep
            time.sleep(0.02)  # 50 Hz control
            
            # Pause physics
            try:
                self.pause_proxy()
            except:
                pass

            self.step_count += 1
            
            # Get observation and compute reward
            obs = self._get_obs()
            reward = self._compute_motorcycle_reward(obs, action)
            done = self._check_motorcycle_termination(obs)
            
            return obs, float(reward), done, self._get_info()

        except Exception as e:
            rospy.logwarn(f"Step error: {e}")
            obs = self._get_obs()
            return obs, -10.0, False, {}

    def _compute_motorcycle_reward(self, obs, action):
        """Reward function based on motorcycle physics"""
        try:
            # Speed-dependent stability reward
            if self.current_speed < self.min_stable_speed:
                # Penalty for being too slow (unstable)
                speed_reward = -20.0 + (self.current_speed / self.min_stable_speed) * 20.0
            else:
                # Reward for good speed
                speed_reward = 10.0 - abs(self.current_speed - 8.0)  # Optimal ~8 m/s
            
            # Dynamic stability reward (most important)
            if self.current_speed > 1.0:
                # At speed, use gyroscopic stability model
                stability_factor = min(self.current_speed / self.min_stable_speed, 2.0)
                balance_reward = 30.0 * stability_factor * math.exp(-5.0 * abs(self.roll))
            else:
                # At low speed, heavily penalize tilt
                balance_reward = -50.0 * abs(self.roll)
            
            # Forward progress reward
            forward_velocity = obs[6] if len(obs) > 6 else 0.0
            progress_reward = 5.0 if forward_velocity > 2.0 else -5.0
            
            # Smooth control reward
            smoothness_penalty = -2.0 * (abs(action[0]) + abs(action[1]))
            
            # Navigation reward
            navigation_reward = 0.0
            if len(obs) > 17:
                current_distance = obs[16]  # distance to goal
                progress = self.prev_distance_to_goal - current_distance
                navigation_reward = 5.0 * progress
                self.prev_distance_to_goal = current_distance
            
            # Steering physics reward (counter-steering at speed)
            steering_reward = 0.0
            if self.current_speed > 5.0 and abs(self.roll) > 0.1:
                # Reward appropriate counter-steering
                expected_steering = -np.sign(self.roll) * min(abs(self.roll), 0.3)
                steering_error = abs(self.steering_angle - expected_steering)
                steering_reward = -5.0 * steering_error
            
            total_reward = (speed_reward + balance_reward + progress_reward + 
                          smoothness_penalty + navigation_reward + steering_reward)
            
            return total_reward
            
        except Exception as e:
            rospy.logwarn_once(f"Reward computation error: {e}")
            return 0.0

    def _check_motorcycle_termination(self, obs):
        """Motorcycle-specific termination conditions"""
        
        # Don't terminate too early
        if self.step_count < 100:
            return False
        
        # Crashed (fallen over)
        if abs(self.roll) > 1.4:  # About 80 degrees
            self.episode_count += 1
            rospy.loginfo(f"Episode {self.episode_count}: Motorcycle crashed! Roll: {self.roll:.2f}")
            return True
        
        # Lost control (spinning)
        if abs(self.yaw_rate) > 3.0 and self.current_speed < 1.0:
            rospy.loginfo(f"Episode {self.episode_count}: Lost control!")
            return True
            
        # Goal reached
        if len(obs) > 16 and obs[16] < 2.0:  # Within 2m of goal
            self.episode_count += 1
            rospy.loginfo(f"Episode {self.episode_count}: Goal reached! Steps: {self.step_count}")
            return True
            
        # Max steps
        if self.step_count >= self.max_steps:
            self.episode_count += 1
            rospy.loginfo(f"Episode {self.episode_count}: Timeout after {self.step_count} steps")
            return True
            
        return False

    def _get_info(self):
        return {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'speed': self.current_speed,
            'roll': self.roll,
            'steering': self.steering_angle
        }

    def reset(self):
        """Reset motorcycle to starting position"""
        try:
            rospy.loginfo("Resetting motorcycle...")
            
            # Reset simulation every few episodes
            if self.episode_count % 3 == 0:
                try:
                    self.reset_simulation_proxy()
                    time.sleep(2.0)
                except:
                    pass
            
            # Reset control state
            self.step_count = 0
            self.throttle = 0.0
            self.steering_angle = 0.0
            self.current_speed = 0.0
            
            # Stop motorcycle
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            
            steering_cmd = Float64()
            steering_cmd.data = 0.0
            self.steering_pub.publish(steering_cmd)
            
            # Give small initial forward velocity for stability
            time.sleep(0.5)
            initial_cmd = Twist()
            initial_cmd.linear.x = 3.0  # Start with some forward speed
            self.cmd_vel_pub.publish(initial_cmd)
            
            # Brief unpause to initialize
            try:
                self.unpause_proxy()
                time.sleep(0.5)
                self.pause_proxy()
            except:
                pass
            
            obs = self._get_obs()
            self.prev_distance_to_goal = obs[16] if len(obs) > 16 else 20.0
            
            return obs
            
        except Exception as e:
            rospy.logwarn(f"Reset error: {e}")
            return self._get_obs()

    def close(self):
        rospy.loginfo("Closing MotorcycleEnv...")
        try:
            # Stop motorcycle
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            
            steering_cmd = Float64()
            steering_cmd.data = 0.0
            self.steering_pub.publish(steering_cmd)
        except:
            pass

if __name__ == '__main__':
    try:
        env = MotorcycleEnv()
        obs = env.reset()
        rospy.loginfo("Testing realistic motorcycle environment...")
        
        for i in range(20):
            # Test realistic motorcycle actions
            throttle = 0.6 if i < 10 else 0.3  # Accelerate then cruise
            steering = 0.1 * math.sin(i * 0.2)  # Gentle weaving
            action = np.array([throttle, steering])
            
            obs, reward, done, info = env.step(action)
            rospy.loginfo(f"Step {i}: Speed={info.get('speed', 0):.1f}m/s, "
                         f"Roll={info.get('roll', 0):.3f}, Reward={reward:.1f}")
            
            if done:
                obs = env.reset()
                
    except rospy.ROSInterruptException:
        pass