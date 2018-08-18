# Range flow odometry

This package implements a range flow and dense visual based odometry estimation method using RGBD image. It works well in dark environments.

<br>

**Related Paper:**

* **Robust Autonomous Flight in Constrained and Visually Degraded Shipboard Environments**, JFR 2017, ICRA 2015, Z. Fang, S. Yang, et al. S. Scherer  [**PDF**](http://www.frc.ri.cmu.edu/~syang/Publications/JFR_2016_ship.pdf)

If you use the code in your research work, please cite the above paper. Please do not hesitate to contact the authors if you have any further questions.

<br>

#### Usage
```catkin_make``` to compile. See ```launch``` file for instructions.

**Subscribe topic:**

RGBD image input: /camera/rgb/image_rect  /camera/depth/image_rect /camera/depth/camera_info


**Output:**

Odometry pose and transform (ros tf) between camera_init and camera


<br>

#### Author & Contacts
Zheng Fang(fangzheng@mail.neu.edu.cn), Shichao Yang(shichaoy@andrew.cmu.edu), Sebastian Scherer(basti@andrew.cmu.edu)
