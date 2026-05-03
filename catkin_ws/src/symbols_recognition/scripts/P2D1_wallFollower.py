#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

class WallFollower:
    def __init__(self):
        rospy.init_node('wall_follower_manually')
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.reset_position(0.0, 0.0)  # Reset the robot to the starting position
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.rate = rospy.Rate(10)  # 10 Hz
        
        self.q_table = np.zeros((3, 3))  # Q-table for 3 states and 3 actions
        self.manual_fill_q_table()  # Fill the Q-table with values
        
    def manual_fill_q_table(self):
        # Manually fill the Q-table with values based on the laser scan data
        self.q_table[0,0] = 100.0  # Turn left
        self.q_table[0,1] = 50.0  # Turn right
        self.q_table[0,2] = 50.0  # Move forward
        self.q_table[1,0] = 50.0  # Turn left
        self.q_table[1,1] = 100.0  # Turn right
        self.q_table[1,2] = 50.0  # Move forward
        self.q_table[2,0] = 50.0  # Turn left
        self.q_table[2,1] = 50.0  # Turn right
        self.q_table[2,2] = 100.0  # Move forward
 
    def get_state(self, data):

        # Determine the current state based on the range angle of laser scan data
        right_distance = min(min(data.ranges[250:300]), 10.0)  # Distance to the right
        left_distance = min(min(data.ranges[70:110]), 10.0)  # Distance to the left
        front_distance = min(min(data.ranges[330:360]), min(data.ranges[0:30]), 10.0)  # Distance to the front
        
        # If there's an obstacle on the front, then Turn left        
        if front_distance < 1.0:
            return 0
        # If there's an obstacle on the right, Turn right
        elif right_distance < 1.0:
            return 1
        # If there's an obstacle on the left, Turn left
        elif left_distance < 1.0:
            return 2
        else: # Move forward
            return  2
        
    def scan_callback(self, data):
        # Process laser scan data to follow the wall
        state = self.get_state(data)
        action = np.argmax(self.q_table[state])  # Choose the action with the highest Q-value
        self.execute_action(action)
        
    def execute_action(self, action):
        msg = Twist()
        if action == 0: # Turn Left
            msg.linear.x = 0.3; msg.angular.z = 0.5
        elif action == 1: # Turn Right
            msg.linear.x = 0.3; msg.angular.z = -0.5
        elif action == 2: # # Move forward
            msg.linear.x = 0.5; msg.angular.z = 0
        self.cmd_pub.publish(msg)
        
    # reset the robot to the starting position
    def reset_position(self, x , y):
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        state = ModelState()
        state.model_name = 'triton'
        state.pose.position.x = x
        state.pose.position.y = y 
    
        set_state(state)
    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
            
if __name__ == '__main__':
    try:
        wall_follower = WallFollower()
        wall_follower.run()
    except rospy.ROSInterruptException:
        pass