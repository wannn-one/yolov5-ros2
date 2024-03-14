#!/usr/bin/env python3
import rclpy
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolo_msgs.msg import BoundingBoxArray

class Republisher(Node):
    def __init__(self):
        super().__init__('inference_republisher')

        self.img_sub = self.create_subscription(Image, '/inference/image_raw', self.img_callback, 10)
        self.obj_sub = self.create_subscription(BoundingBoxArray, '/inference/object_raw', self.obj_callback, 10)

    def img_callback(self, msg):
        cv_bridge = CvBridge()
        img = cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow('YOLOv5', img)
        cv2.waitKey(1)

    def obj_callback(self, msg):
        print(msg)

def main(args=None):
    rclpy.init(args=args)
    republisher = Republisher()
    rclpy.spin(republisher)
    republisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()