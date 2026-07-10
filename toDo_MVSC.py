
请帮我写一个mvsc的数据集用于训练和测试，参考EAG3R，
.:
outdoor_day  outdoor_night  README_EAG3R.txt

./outdoor_day:
outdoor_day2_data.bag  outdoor_day2_gt.bag

./outdoor_night:
outdoor_night1_data.bag  outdoor_night2_data.bag  outdoor_night3_data.bag
outdoor_night1_gt.bag    outdoor_night2_gt.bag    outdoor_night3_gt.bag


每个bag的文件格式如下：
ROS Bag Data Format#
Each sequence consists of a data ROS bag, with the following topics:

/davis/left/events (dvs_msgs/EventArray) - Events from the left DAVIS camera.
/davis/left/image_raw (sensor_msgs/Image) - Grayscale images from the left DAVIS camera.
/davis/left/imu (sensor_msgs/Imu) - IMU readings from the left DAVIS camera.
/davis/right/events (dvs_msgs/EventArray) - Events from the right DAVIS camera.
/davis/right/image_raw (sensor_msgs/Image) - Grayscale images from the right DAVIS camera.
/davis/right/imu (sensor_msgs/Imu) - IMU readings from the right DAVIS camera.
/velodyne_point_cloud (sensor_msgs/PointCloud2) - Point clouds from the Velodyne (one per spin).
/visensor/left/image_raw (sensor_msgs/Image) - Grayscale images from the left VI-Sensor camera.
/visensor/right/image_raw (sensor_msgs/Image) - Grayscale images from the right VI-Sensor camera.
/visensor/imu (sensor_msgs/Imu) - IMU readings from the VI-Sensor.
/visensor/cust_imu (visensor_node/visensor_imu) - Full IMU readings from the VI-Sensor (including magnetometer, temperature and pressure).
Two sets of custom messages are used, dvs_msgs/EventArray from rpg_dvs_ros and visensor_node/visensor_imu from visensor_node. The visensor_node package is optional if you do not need the extra IMU outputs (magnetometer, temperature and pressure.

In addition, each corresponding ground truth bag contains the following topics:

/davis/left/depth_image_raw (sensor_msgs/Image) - Depth maps for the left DAVIS camera at a given timestamp (note, these images are saved using the CV_32FC1 format (i.e. floats).
/davis/left/depth_image_rect (sensor_msgs/Image) - Rectified depth maps for the left DAVIS camera at a given timestamp (note, these images are saved using the CV_32FC1 format (i.e. floats).
/davis/left/blended_image_rect (sensor_msgs/Image) - Visualization of all events from the left DAVIS that are 25ms from each left depth map superimposed on the depth map. This message gives a preview of what each sequence looks like.
/davis/left/odometry (geometry_msgs/PoseStamped) - Pose output using LOAM. These poses are locally consistent, but may experience drift over time. Used to stitch point clouds together to generate depth maps.
/davis/left/pose (geometry_msgs/PoseStamped) - Pose output using Google Cartographer. These poses are globally loop closed, and can be assumed to have minimal drift. Note that these poses were optimized using Cartographer’s 2D mapping, which does not optimize over the height (Z axis). Pitch and roll are still optimized, however.
/davis/right/depth_image_raw (sensor_msgs/Image) - Depth maps for the right DAVIS camera at a given timestamp.
/davis/right/depth_image_rect (sensor_msgs/Image) - Rectified depth maps for the right DAVIS camera at a given timestamp.
/davis/right/blended_image_rect (sensor_msgs/Image) - Visualization of all events from the right DAVIS that are 25ms from each right depth map superimposed on the depth map. This message gives a preview of what each sequence looks like