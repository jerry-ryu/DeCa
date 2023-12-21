#! /usr/bin/env python3
import cv2

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CameraNode:
    def __init__(self):
        rospy.init_node('camera_publisher', anonymous=True)
        self.pub = rospy.Publisher('lane_image', Image, queue_size=10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    
    def filter(self, frame):
        ROI = frame[70:,:].copy()
        gray_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_img, (9,9), sigmaX=1.2, sigmaY=1.2)
        canny_img = cv2.Canny(blur_img, 200, 350, apertureSize=3)
        
        return canny_img
    
    def publish_frame(self):
        ret, frame = self.cap.read()
        if ret:
            filtered = self.filter(frame)
            cv2.imshow('Webcam Feed',filtered)
            ros_image = self.bridge.cv2_to_imgmsg(filtered, "mono8")
            self.pub.publish(ros_image)
    
    def run(self):
        while not rospy.is_shutdown():
            self.publish_frame()

        self.cap.release()
        #cv2.destroyAllWindows()

if __name__ == '__main__':
    camera_node = CameraNode()
    camera_node.run()