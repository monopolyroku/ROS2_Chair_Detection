# Chair Detection with ROS2 (~~Initially coined SeatOccupancyDetection~~)

## Purpose
The main purpose of this repository is to store code that investigates the possibility of implementing object detection technologies in the SIT Library @ Punggol slated to open in 2024. 

The technologies used in this project are as follows:
- Tensorflow
- Tensorflow Lite
- OpenCV (Open Computer Vision)
- ROS2 (Robotics Operating System 2)


## Tensorflow implementation

#### Pre-trained model: 
There are two options whenever it comes to object detection models, doing end-to-end training to create a new model or using pre-trained models. As the time and resources available were not available to implement end-to-end training, the pre-trained model route was preferred, especially since there is a [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) provided by tensorflow, a free and open-source software library for machine learning and artificial intelligence.

##### Faster R-CNN Inception V2

Based on a [initial project](https://github.com/RexxarCHL/library-seat-detection) done by github user RexxarCHL, the [Faster R-CNN Inception v2 trained on COCO dataset](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) was recommended as it had higher detection consistency and faster computation speed compared to several pre-trained models being used (specifically Mobilenet v2, Mask R-CNN)

Inception v2 is a convolutional neural network architecture that was introduced in 2016, it uses a combination of 1x1 3x3 and 5x5 convolutions along with pooling layers to capture features at different scales. 
The architecture also includes batch normalization and residual connections to improve training and performance, Faster R-CNN algorithm combines Inception v2 architecture with a region proposal network and a classifier to perform object detection. 
The RPN generates proposals for potential object locations, followed by a second stage for classification and bounding box regression. 


The following files:
- multiprocess_object_detect.py
- object_detector.py
- simple_object_detect.py
- v0_multithread.py

contains experimental code to test and familiarize the functions required to perform object detection. Multithreading and Multiprocessing was attempted on the original forked code to speed up the FPS (Frames per second) however multiprocessing without RTOS (Real-Time Operating System) implementation proved futile. 
Although, multithreading is possible there is not much benefit to it as the FPS remained consistent at approximately 1-2 FPS. 
Ater the disappointing episode, other lightweight alternatives were considered to increase the speed and peformance of the Chair detection algorithm.


## Tensorflow Lite and ROS2 implementation
Tensorflow lite is a set of tools that enables on-device machine learning by allowing developers to run their models on mobile devices. This means that computational requirements were not as intensive as the original Tensorflow while achieving almost equal performance. 
Although most pre-trained models can be converted to run with Tensorflow Lite, the recommended ones only have direct prediction of the bounding box offset and class scores from feature maps at a single stage which increases the FPS (Frames per second) of the program as the models are much simpler.

For this project, the Tensorflow Lite implementation forked from this [repo](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi) is integrated with ROS2 to publish the information (pose, empty chair lot etc.) from a CCTV camera to the robot.

However, do note that running the same program on different hardware will affect the latency/FPS of the program as seen on the top left hand corner of the images below.

On Raspberry Pi 3 Model B V1.2: <br>
<img src="https://i.postimg.cc/7Pz0n70H/rpi-raspios.jpg" alt="Slower FPS" width="240" height="203"/><br>


On Lenovo Ideapad 5: <br>
<img src="https://i.postimg.cc/kMWRbCPD/ideapad-ubuntu20-04.jpg" alt="Slower FPS" width="240" height="212"/><br>

## Videos: Tensorflow Faster R-CNN Inception V2 and Tensorflow Lite Efficient 

<a href="https://www.youtube.com/watch?v=pLGPopefdiE" target="_blank"><img src="https://i9.ytimg.com/vi/pLGPopefdiE/mqdefault.jpg?sqp=COjomqAG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGFogWihaMA8=&rs=AOn4CLD8h9p6HPTPTYsvTzjag-TW_140sw" 
alt="Tensorflow: Faster R-CNN Inception V2" width="240" height="180" border="10" /></a>

<a href="https://www.youtube.com/watch?v=SgjVN6L3r1k" target="_blank"><img src="https://i9.ytimg.com/vi/SgjVN6L3r1k/mqdefault.jpg?sqp=CJTrmqAG-oaymwEmCMACELQB8quKqQMa8AEB-AHUBoAC4AOKAgwIABABGH8gEyh6MA8=&rs=AOn4CLCAnZDG6dOWbvKGjlPSDhBcWRI8_A" 
alt="Tensorflow Lite: Efficient" width="240" height="180" border="10" /></a>


## How to build and run ROS2 src folder 
Run the set of instructions in the README.md in this [repo](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi) preferably in a Linux environment, make sure everything works before moving on to the next step.

1. Place the src folder in a new directory, ensure that ROS2 is initialised in that folder
2. Under function *main()* edit the '--model' default path to the the efficientdet_lite0.tflite in your personal system
3. Use ```colcon build ``` to build the package within the src folder, creating the install, build and log folders in the process
4. For good measure ``` source ~/.bashrc ``` and ``` source install/setup.bash ``` to refresh the terminal bash file and setup bash file after the new build
5. Use the command ``` ros2 run chair_detect chair_detect ``` to run the chair_detect node and a window should pop up showing your camera's pov


## To take note of during integration
To be updated.