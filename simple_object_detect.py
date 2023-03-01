# Author: monopolyroku (colin)
# Code has been modified by me to utilize the threading library to serialize the data from the WebCam
# For more details refer to the README.md file in the github repo

# Code apapted from tensorflow-human-detection by madhawav and library-seat-detection by RexxarCHL
# https://github.com/RexxarCHL/library-seat-detection

import numpy as np
import tensorflow as tf
import cv2
import time
import threading as thread
from collections import namedtuple

CvColor = namedtuple('CvColor', 'b g r')
BLUE = CvColor(255, 0, 0)
RED = CvColor(0, 0, 255)


class ObjectDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.detection_graph = tf.Graph()

        # Loads the frozen model
        with self.detection_graph.as_default():
            frozen_graph = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.model_path, 'rb') as f:
                serialized_graph = f.read()
                frozen_graph.ParseFromString(serialized_graph)
                tf.import_graph_def(frozen_graph, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Define input and output tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (
                int(boxes[0, i, 0] * im_height),
                int(boxes[0, i, 1] * im_width),
                int(boxes[0, i, 2] * im_height),
                int(boxes[0, i, 3] * im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        # self.default_graph.close()



# Created a class solely for video input and multithreading so that detection will be much smoother

class WebCamStream:
    def __init__(self, src=0):
        # initialized the webcam stream and read the first frame from it 
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialized variable used to indicate if thread should be stopped
        self.stopped = False

    def start(self):
        # Places the update method in a separate thread from main python script
        # thus reducing latency and increasing FPS 
        thread.Thread(target=self.update, args=()).start()
        return self   

    def update(self):
        # keep looping till thread is stopped 
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()       

    def read(self):
        # returns frame most recently read
        return self.frame

    def stop(self):
        # indicates thread should be stopped 
        self.stopped = True



if __name__ == "__main__":
    # Using simpler object detection model
    model_path = 'C:\\Users\\colin\\OneDrive\\Desktop\\RSE\\Y2T2\\Project 4\\Seat Occupancy Detection\\github\\library-seat-detection\\ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03\\frozen_inference_graph.pb'
    odapi = ObjectDetector(model_path)
    threshold = 0.65

    # starts a live video feed but faster (hopefully)
    vs = WebCamStream(src=0).start()


    # Creates a frame of above defined values for video
    result = cv2.VideoWriter('sod.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, (640,480))


    while True:
        # success, img = cap.read()
        frame = vs.read()

        boxes, scores, classes, num = odapi.processFrame(frame)

        for i in range(len(boxes)):
            box = boxes[i]
            if classes[i] == 62 and scores[i] > threshold:  # Chair
                cv2.putText(
                    frame, "chair: {}".format(scores[i]),
                    (box[1]+10, box[0]+10), cv2.FONT_HERSHEY_PLAIN,
                    0.9, RED)
                cv2.rectangle(
                    frame,
                    (box[1], box[0]),
                    (box[3], box[2]),
                    RED, 2)


        # cv2.imshow("preview", frame)

        # Writes processed frame into the file sod.mp4
        result.write(frame)


        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # releases the frame
    odapi.close()
    vs.stop()
    cv2.destroyAllWindows()

