#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool

class BoolPublisher:
    def __init__(self):
        rospy.init_node('reset_node', anonymous=True)
        self.publisher = rospy.Publisher('reset', Bool, queue_size=10)

    def run(self):
        while not rospy.is_shutdown():
            user_input = input("Enter 1 to publish True: ") 
            
            if user_input == '1':
                self.publisher.publish(True)
                rospy.loginfo("Published True on /bool_topic")

if __name__ == '__main__':
    publisher = BoolPublisher()
    publisher.run()
