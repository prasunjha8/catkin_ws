# RL Self-Balancing Motorcycle in ROS & Gazebo

This project demonstrates how to train a reinforcement learning (RL) agent to control a two-wheeled, self-balancing motorcycle in the Gazebo simulator. The agent's goal is to learn both self-stabilization and autonomous navigation to a target waypoint.

The project is built within the **ROS (Robot Operating System)** framework and uses a custom **OpenAI Gym environment** to interface with the Gazebo physics simulator. The RL agent is implemented using the **Stable-Baselines3** library with the **Soft Actor-Critic (SAC)** algorithm.

---

## Features
- **Realistic Simulation**: A dynamically unstable two-wheeled robot model (URDF) simulated in Gazebo 11.
- **ROS Integration**: Fully integrated with ROS Noetic, using standard ROS topics and services for communication.
- **Custom RL Environment**: An OpenAI Gym `gym.Env` interface (`bike_env.py`) bridging the RL agent and the Gazebo simulation.
- **Advanced RL Agent**: Uses the SAC algorithm from Stable-Baselines3 for efficient learning in a continuous action space.
- **Complete Workflow**: Includes scripts for training (`train.py`) and testing (`test_model.py`).

---

## Project Structure
```
/catkin_ws
└── /src
    ├── /two_wheel_bike_description
    │   ├── /urdf
    │   │   └── bike.urdf.xacro
    │   ├── /worlds
    │   │   └── training_world.world
    │   ├── /launch
    │   │   └── spawn_bike.launch
    │   ├── CMakeLists.txt
    │   └── package.xml
    │
    └── /bike_rl_agent
        ├── /scripts
        │   ├── train.py
        │   ├── bike_env.py
        │   └── test_model.py
        ├── CMakeLists.txt
        └── package.xml
```

---

## Prerequisites & Setup
This project is designed to run inside a **Docker container** to ensure a consistent environment, especially on non-native Linux systems like macOS (M1/M2) or Windows.

### Requirements
- Docker Desktop

### Installation and Setup Guide

#### 1. Clone the Repository
Clone this repository to your local machine.

#### 2. Launch the Docker Container
Run the following command on your host machine:
```bash
docker run -it --rm --platform linux/amd64 -p 6080:80 -p 5900:5900 --shm-size=512m tiryoh/ros-desktop-vnc:noetic
```
Access the container desktop environment:
- VNC: `localhost:5900`
- Browser: [http://localhost:6080/](http://localhost:6080/)

#### 3. Copy Project Files into the Container
Find your container’s name:
```bash
docker ps
```
Copy your project files:
```bash
docker cp /path/to/your/catkin_ws/src [YOUR_CONTAINER_NAME]:/home/ubuntu/catkin_ws/
```

#### 4. Setup Inside the Container
Run the following commands inside the container:

**a. Fix File Ownership:**
```bash
sudo chown -R ubuntu:ubuntu ~/catkin_ws
```

**b. Install Python Dependencies:**
```bash
pip3 install --user torch stable-baselines3==1.8.0 rospkg
```

**c. Build the Catkin Workspace:**
```bash
cd ~/catkin_ws
catkin_make
```

**d. Source the Workspace:**
```bash
source devel/setup.bash
```

**e. Make Scripts Executable:**
```bash
chmod +x ~/catkin_ws/src/bike_rl_agent/scripts/*.py
```

---

## Usage
The training and simulation require **two separate terminals** inside the container.

### Terminal 1: Launch Gazebo
```bash
source ~/catkin_ws/devel/setup.bash
roslaunch two_wheel_bike_description spawn_bike.launch
```
This opens Gazebo with the motorcycle on a flat plane.

### Terminal 2: Run Training/Testing

**To Train a New Agent:**
```bash
source ~/catkin_ws/devel/setup.bash
rosrun bike_rl_agent train.py
```
The agent learns, with progress shown in the terminal. Models and checkpoints save to `/tmp/gym/`.

**To Test a Trained Agent:**
```bash
source ~/catkin_ws/devel/setup.bash
rosrun bike_rl_agent test_model.py
```
This loads the trained model and runs test episodes, printing results and showing the bike’s behavior in Gazebo.

---

