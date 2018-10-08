# PATH\_PLANNING\_2D


The `PATH_PLANNING_2D` package implements the QV-Tree Search method proposed in "Stochastic 2-D Motion Planning with a POMDP Framework". The algorithm solves the stochastic 2-D motion planning problem considering noise from both the robot dynamics and sensor measurements.

A simulator is included to demonstrate the behavior of the proposed motion planner. The package also contains implementation of methods like A* and MDP for comparison purposes.

The software is tested on Ubuntu 16.04 with ROS Kinetic.

Video: [https://youtu.be/_T7Rt0iTyJQ](https://youtu.be/_T7Rt0iTyJQ)<br/>
Paper Draft: [https://arxiv.org/abs/1810.00204](https://arxiv.org/abs/1810.00204)

## License

GNU General Public License v3.0

## Dependencies

Most of the dependencies are standard including `Eigen`, `OpenCV`, and `Boost`. The standard shipment from Ubuntu 16.04 and ROS Kinetic works fine. One of the special requirement is [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive) for acceleration of compution. The package also depends on [JPS3d](https://github.com/KumarRobotics/jps3d) for the A* implementation.

## Compling
The software is a standard catkin package. Make sure the package is on `ROS_PACKAGE_PATH` after cloning the package to your workspace. And the normal procedure for compiling a catkin package should work.

```
cd your_work_space
catkin_make --pkg path_planning_2d --cmake-args -DCMAKE_BUILD_TYPE=Release
```
