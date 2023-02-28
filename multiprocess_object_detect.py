

# Author: monopolyroku (colin)
# Code has been modified by me to utilize the threading library to serialize the data from the WebCam
# For more details refer to the README.md file in the github repo

# Code apapted from tensorflow-human-detection by madhawav and library-seat-detection by RexxarCHL
# https://github.com/RexxarCHL/library-seat-detection

import numpy as np
import tensorflow as tf
import cv2
import time
import multiprocessing as mp 
from collections import namedtuple

CvColor = namedtuple('CvColor', 'b g r')
# BLUE = CvColor(255, 0, 0)
# GREEN = CvColor(0, 255, 0)
# ORANGE = CvColor(0, 128, 255)
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

        # Code below uses numpy broadcasting to apply scaling factor to all elements at once
        # then using astype to cast an array to a spefied data type in this case an integer
        boxes_list = np.array(boxes[0, :boxes.shape[1], :4] * np.array([im_height, im_width, im_height, im_width])).astype(int)

        # Returns the output of the object detection algo for a single input frame
        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        # self.default_graph.close()



# Created a class solely for video input and multithreading so that detection will be much smoother

class WebCamStream:
    def __init__(self, src=0):
        self.src = src
        # initialized variable used to indicate if thread should be stopped
        self.stopped = False
        # initialize frame to None 
        self.frame = None
        # initialize a multiprocessing lock
        self.lock = mp.Lock()

    def start(self):
        # creates a new multiprocessing process specifies that target of the process should be update method 
        p = mp.Process(target=self.update, args=())
        # process will run as a daemon process and automatically terminate when main program finishes
        p.daemon = True
        p.start()
        return self   

    def update(self):
        # keep looping till thread is stopped 
        # reads frames from camera and stores in self.frame
        # sets self.grabbed flag to indicate new frame is available
        # allows read method to efficiently retrieve it as soon as its available 
        stream = cv2.VideoCapture(self.src)

        while True:
            if self.stopped:
                stream.release()
                return
            
            (grabbed, frame) = stream.read()
            if grabbed:
                with self.lock:
                    self.frame = frame
                #(self.grabbed, self.frame) = self.stream.read()       

    def read(self):
        # returns frame most recently read
        while self.frame is None:
            pass
        # returns the most recently read frame 
        with self.lock:
            frame = self.frame
        return self.frame

    def stop(self):
        # indicates thread should be stopped 
        self.stopped = True



if __name__ == "__main__":
    model_path = 'C:\\Users\\colin\\OneDrive\\Desktop\\RSE\\Y2T2\\Project 4\\Seat Occupancy Detection\\github\\library-seat-detection\\faster_rcnn_inception_v2_coco_2018_01_28\\frozen_inference_graph.pb'
    odapi = ObjectDetector(model_path)
    threshold = 0.65

    # starts a live video feed but faster (hopefully)
    vs = WebCamStream(src=0)
    vs.start()

    # Creates a frame of above defined values for video
    result = cv2.VideoWriter('sod.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, (640,480))

    # Initialise the amount of processors being used by the .pool function to 4
    num_process = 4
    pool = mp.Pool(num_process)


    while True:
        # success, img = cap.read()
        frame = vs.read()

        # apply ProcessFrame() method to each frame in parallel using multiprocessing
        # creates a list 'results' by applying odapi.processFrame method to frame object in parallel using pool map method
        # map method applies function to a sequence of arguments and returns a list of results of function applied to each argument
        # odapi.processFrame takes a single argument image (frame from webcam stream) and returns a tuple containing info about object detected
        results = pool.map(odapi.processFrame, [frame])

        boxes, scores, classes, num = results[0]

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


        # Writes processed frame into the file sod.mp4
        result.write(frame)

        # cv2.imshow("preview", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break


    # releases the frame
    result.release()
    # .close tells pool not to accept any new jobs
    pool.close()
    # .join tells pool to wait until all jobs finished then exit, cleaning up the pool
    pool.join()
    odapi.close()
    vs.stop()
    cv2.destroyAllWindows()

