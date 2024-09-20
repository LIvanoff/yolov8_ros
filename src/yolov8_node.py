#!/usr/bin/env python3
import cv2
import rospy

from cv_bridge import CvBridge

from ultralytics import YOLO

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from detection_msgs.msg import BoundingBox, BoundingBoxes

IMG_ORIGIN_W = 2448
IMG_ORIGIN_H = 2048

class Yolov8Node:

    def __init__(self):
        self.input_image_topic = rospy.get_param("~input_image_topic", "/camera/center/image_raw/compressed")
        self.weights = rospy.get_param("~weights", "best.pt")
        self.output_detections_topic = rospy.get_param("~output_detections_topic", "/yolov8/detections")
        self.output_array_topic = rospy.get_param("~output_array_topic", "/yolov8/array")

        self.model = YOLO(self.weights)

        self.model.to("cuda:0")
        self.threshold = 0.5
        self.cv_bridge = CvBridge()

        # topcis
        self._pub_detect = rospy.Publisher(self.output_detections_topic, BoundingBoxes, queue_size=10)
        self._pub_array = rospy.Publisher(self.output_array_topic, Float32MultiArray, queue_size=10)
        self._sub = rospy.Subscriber(self.input_image_topic, CompressedImage, self.image_cb)
        print('Initialization is complete')

        rospy.spin()

    def image_cb(self, msg: CompressedImage):
        img = self.cv_bridge.imgmsg_to_cv2(msg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=img, show=False)

        # create detections msg
        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = msg.header
        bounding_boxes.image_header = msg.header

        float_array = Float32MultiArray()

        results = results[0].cpu()

        for b in results.boxes:

            label = self.model.names[int(b.cls)]
            score = float(b.conf)

            if score < self.threshold:
                continue

            bounding_box = BoundingBox()

            box = b.xywh[0]


            bounding_box.Class = label
            bounding_box.probability = b.conf
            bounding_box.xmin = float(box[0])
            bounding_box.ymin = float(box[1])
            bounding_box.xmax = float(box[2])
            bounding_box.ymax = float(box[3])

            # add data in FloatArray
            for value in box:
                float_array.data.append(value)

            bounding_boxes.bounding_boxes.append(bounding_box)
        self._pub_detect.publish(bounding_boxes)
        self._pub_array.publish(float_array)

if __name__ == "__main__":
    try:
        rospy.init_node("yolov8_node")
        node = Yolov8Node()
    except rospy.ROSInternalException:
        exit()

