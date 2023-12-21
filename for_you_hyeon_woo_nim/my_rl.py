#! /usr/bin/env python3

import rospy
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Bool
#=========================

import time

# For Image ==============
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
#=========================

import gym
from gym import spaces
import numpy as np
    

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(16*51)  # Replace N with the number of actions
        # Example for using image as input:
        # Grayscale image observation (280x640, result of cv2.Canny)
        image_space = spaces.Box(low=0, high=1, shape=(280, 640, 1), dtype=np.uint8)
        # Speed value (assuming a scalar between a defined min and max)
        speed_space = spaces.Box(low=1520, high=1535, shape=(1,), dtype=np.uint32)
        # Steering value (assuming a scalar between a defined min and max)
        steering_space = spaces.Box(low=65, high=115, shape=(1,), dtype=np.uint32)
        
        mission_space = spaces.Discrete(5)

        self.observation_space = spaces.Tuple((image_space, speed_space, steering_space, mission_space))

        rospy.init_node('bran_node',anonymous=False)

        self.pub=rospy.Publisher('action', Int32MultiArray, queue_size=10)

        self.bridge=CvBridge()
        self.speed=1520
        self.steering=90
        self.image=None
        self.mission=0
        self.lane_out=False

        self.init_ros()
    
    def step(self, action):
        # action = [drive, steering]
        # Execute one time step within the environment
        # Replace the following with your own logic

        # 1. Perform action.
        int32_array_msg = Int32MultiArray()
        int32_array_msg.data =action
        self.pub.publish(int32_array_msg)
        time.sleep(0.5)

        # 2. Update Observation
        
        observation = ...

        # Check the Arduino light sensor value
        # Assuming you have a method to get the sensor value
        # Set reward based on sensor value
        if self.lane_out == True:
            # Penalize for stepping on the black line
            reward = -1



        done = ...
        info = {}
        return observation, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        # Replace the following with your own logic
        observation = ...
        return observation

    
    def init_ros(self):
        rospy.Subscriber('vehicle_state', Int32MultiArray, self.vehicle_state_callback)
        rospy.Subscriber('lane_out',Bool, self.lane_out_callback)
        rospy.Subscriber('reset',Bool, self.reset_callback)
        rospy.Subscriber('lane_image',Image, self.image_callback)

    def image_callback(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print(e)
    
    def vehicle_state_callback(self,data):
        self.speed=data.data[0]
        self.steering=data.data[1]

    def lane_out_callback(self,data):
        self.lane_out = data.data
    
    def reset_callback(self,data):
        self.reset=data.data