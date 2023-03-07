# Creates a list that will contain detected chairs 
# currently contains fake data
detected_chairs = [0, 1, 0, 1]



# Class definition for ROS2 publisher
class ChairDetector(Node):
  
  def __init__(self):
    super().__init__('chair_detector')
    self.publisher_ = self.create_publisher(Int8MultiArray, 'sod_topic', 10)
    timer_period = 0.5 
    self.timer = self.create_timer(timer_period, self.publishing)
    self.i = 0
  
  def publishing(self): 
    # Publish the coordinates of the detected chairs to a ROS topic
    point_msg = Int8MultiArray()
    # Assign values from detected_chairs list to point_msg
    point_msg.data = detected_chairs
    self.publisher_.publish(point_msg)
    self.get_logger().info('Publishing: "%s"' % point_msg.data)
    self.i += 1





 # Initialization of ROS2 node
  rclpy.init()
  chair_detector = ChairDetector()
  rclpy.spin(chair_detector)
  # Destroys ROS2 node
  chair_detector.destroy_node()
  rclpy.shutdown()