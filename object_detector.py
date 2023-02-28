import numpy as np
import tensorflow as tf
import cv2
import time
from collections import namedtuple


CvColor = namedtuple('CvColor', 'b g r')
BLUE = CvColor(255, 0, 0)
GREEN = CvColor(0, 255, 0)
RED = CvColor(0, 0, 255)
ORANGE = CvColor(0, 128, 255)

# Code apapted from tensorflow-human-detection by madhawav and library-seat-detection by RexxarCHL
# https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2
# https://github.com/RexxarCHL/library-seat-detection

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


if __name__ == "__main__":
    model_path = 'C:\\Users\\colin\\OneDrive\\Desktop\\RSE\\Y2T2\\Project 4\\Seat Occupancy Detection\\github\\library-seat-detection\\faster_rcnn_inception_v2_coco_2018_01_28\\frozen_inference_graph.pb'
    odapi = ObjectDetector(model_path)
    threshold = 0.65

    # input via a video contained within the library-seat-detection folder 
    # cap = cv2.VideoCapture('testVid6.mp4')

    # input via a live video feed
    cap = cv2.VideoCapture(0)

    # seats = [[(390, 486), (742, 765)], [(921, 494), (1234, 715)], [(828, 276), (1140, 495)], [(1117, 289), (1392, 479)]]

    while True:
        success, img = cap.read()
        if not success:
            break
        
        # resizes preview screen to 1280 x 720 pixels
        img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.
        # Draws bounding boxes around the set coordinates from the seats array above 
        # for i, seat in enumerate(seats):
        #     (x0, y0), (x1, y1) = seat[0], seat[1]
        #     cv2.putText(img, "seat{}".format(i), (x0+10, y0+10), cv2.FONT_HERSHEY_PLAIN, 0.9, RED)
        #     cv2.rectangle(img, (x0, y0), (x1, y1), RED, 2)

        for i in range(len(boxes)):
            box = boxes[i]
            if classes[i] == 1 and scores[i] > threshold:  # Human
                cv2.putText(
                    img, "human: {}".format(scores[i]),
                    (box[1]+10, box[0]+10), cv2.FONT_HERSHEY_PLAIN,
                    0.9, BLUE)
                cv2.rectangle(
                    img,
                    (box[1], box[0]),
                    (box[3], box[2]),
                    BLUE, 2)
            elif classes[i] == 62 and scores[i] > threshold:  # Chair
                cv2.putText(
                    img, "chair: {}".format(scores[i]),
                    (box[1]+10, box[0]+10), cv2.FONT_HERSHEY_PLAIN,
                    0.9, ORANGE)
                cv2.rectangle(
                    img,
                    (box[1], box[0]),
                    (box[3], box[2]),
                    ORANGE, 2)
            elif scores[i] > threshold:
                cv2.putText(
                    img, "obj{}: {}".format(classes[i], scores[i]),
                    (box[1]+10, box[0]+10),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.9, RED)
                cv2.rectangle(
                    img,
                    (box[1], box[0]),
                    (box[3], box[2]),
                    RED, 2)

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
