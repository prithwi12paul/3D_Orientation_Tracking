# 3D_Orientation_Tracking


## Project Overview

This project estimates the 3-D orientation of a rigid rotating body over time using the IMU angular velocity and linear acceleration measurements. It is important to
accurately track a robot's pose in a dynamic or static environment for executing essential tasks such as collision avoidance, safe navigation, and control. Additionally, motion tracking is important for carrying out tasks that require precise alignment and positioning, for example, assembly and inspection 

In the first phase of the project, the IMU calibration has been performed where the raw ADC data have been converted to physical units by using the appropriate scale factor and sensitivity of the gyroscope and the accelerometer and then bias correction has been performed. Next, the quaternion trajectory is predicted using the quaternion motion model and the IMU angular velocity measurements. The predicted acceleration is computed using the observation model where we know that since the body is rotating, the linear acceleration would be only 'g' in the z direction. In order to minimize the drift in the trajectory and ensure smoothness in the motion, a constrained gradient descent algorithm has been implemented to minimize the cost function and estimate the orientation from IMU measurements 

In the second part of the project, a panoramic image has been generated by stitching the camera images obtained by the rotating body using the optimized orientation estimates

## Project File Structure

### Datasets

The [data](https://drive.google.com/file/d/1fijuFjKSXvZfbPsZya0OdWZUJkB45_9J/view?usp=share_link) contains the dataset (pickle files) which has been collected by the graduate student researchers in the Existential Robotics Laboratory, University of California San Diego. There would be total 11 datasets and our model is tested on these datasets. It contains the raw IMU data of acceleration and angular velocity measurements, the euler angles from the VICON motion capture system and the RGB pixel coordinates. Data synchronization has been performed to map the data having matching timestamps and then used for further analysis.

### Source Code

Necessary Python Libraries required to run the code

NUMPY, JAX, JAXLIB, TRANSFORMS3D

#### Necessary Python Libraries

The third party modules used are as listed below. They are included as [`requirements.txt`](requirements.txt).

- ipython==7.31.1
- matplotlib==3.5.2
- numpy==1.21.5
- jax==0.4.2python3 Orientation_Tracking.pypython3 Orientation_Tracking.py
- pandas==1.5.3
- transforms3d==0.4.1


Python files

- [load_data.py](load_data.py) - Loads the pickle files present in the `data/` folder
- [Orientation_Tracking.py](Orientation_Tracking.py) - Main python file that runs the Orientation Tracking PGD code

### Jupyter Notebook

The [Jupyter Notebook](Orientation_Tracking.ipynb)is the notebook version of the main python code which has all the visualization and plots for the results

## How to run the code

Install all required libraries -

```
pip install -r requirements.txt

```
Run the load_data.py file -

```
python3 load_data.py
```

Main source code -

- Mention the datafile path in the 'imu', 'vicon' and 'cam_data variable'
- Specify the value of 'n_samples' variable according to the dataset given (Instructions to set the value is given in the comments section of the code) Default value is 700.
- Specify the number of iterations in the gradient descent step in the 'n_iters' variable. By default, the value is 10.

- Run the Orientation_Tracking.py file

```
python3 Orientation_Tracking.py
```











