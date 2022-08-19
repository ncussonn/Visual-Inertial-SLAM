SCRIPTS
############

main.py:	contains the main script, running this file with pr3_utils in same
		directory will complete steps a through c of the project

pr3_utils.py:	contains all the imported functions used for the project 

###################################
##### pr3_utils FUNCTIONS #########	
###################################

load_data:	

function to read visual features, IMU measurements and calibration parameters
	Input:
		file_name: the input data file. Should look like "XX.npz"
	Output:
		t: time stamp
			with shape 1*t
		features: visual feature point coordinates in stereo images, 
			with shape 4*n*t, where n is number of features
		linear_velocity: velocity measurements in IMU frame
			with shape 3*t
		angular_velocity: angular velocity measurements in IMU frame
			with shape 3*t
		K: (left)camera intrinsic matrix
			with shape 3*3
		b: stereo camera baseline
			with shape 1
		imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
			with shape 4*4

hatMapSetup:

	Computes vectorized hat maps for code.
	Input:
		t: time stamp
			with shape 1*t
		V: linear velocity
			with shape 3*t
		w: angular velocity
			with shape 3*t		
	Output:
		u_hat:			4*4*t matrix for u hat
		u_curly_hat:		6*6*t matrix for u curly hat
		v_hat:			3*3*t matrix for v hat
		w_hat:			3*3*t matrix for w hat

prediction:

	Computes pose of robot over time assuming there is no noise.
	Input:
		t: time stamp
			with shape 1*t
		V: linear velocity
			with shape 3*t
		w: angular velocity
			with shape 3*t
		W: motion noise
			with shape 6*6
	Output:
		mu: pose of robot (IMU considered origin)
		   with shape 4*4*N
		Sigma: covariance of Sigma for IMU pose
			with shape 6*6*N

update:

	function to do Visual Mapping via the EKF update
	Input:
		features: 	4*M*T matrix representing selected features
					for all time in image frame
		M:		number of features
		T:		number of timesteps t
		K:		intrinsic camera calibration matrix 4*4
		b:		baseline - scalar
		pose:		extrinsic transform from imu to world 4*4*T in SE(3)
		imu_T_cam:	extrinsic transform from camera to imu 4*4 in SE(3)
		v:		observation noise
		
	Output:
		mu:		4*M matrix representing final mean map feature coordinates in world frame
		Sigma:		3M*3M matrix representing observation covariances

viSLAM:

	Function to perform the VI-SLAM part of the project
	Input:
		features: 	4*M*T matrix representing selected features
					for all time in image frame
		M:			number of features
		T:			number of timesteps t
		K:			intrinsic camera calibration matrix 4*4
		b:			baseline - scalar
		pose:		extrinsic transform from imu to world 4*4*T in SE(3)
		imu_T_cam:	extrinsic transform from camera to imu 4*4 in SE(3)
		v:			observation noise
		time_stamp: time stamp with shape 1*t			
		V: 			linear velocity with shape 3*t			
		w: 			angular velocity with shape 3*t
		W: 			motion noise with shape 6*6

	Output:
		mu_pose: mean pose of robot (IMU considered origin)
		   		with shape 4*4*T
		mu_map:	mean map of landmarks after final timestep
		Sigma: covariance of Sigma for IMU pose and each feature
			with shape 3M+6*3M+6

visualize_trajectory_2d:

	modified function to visualize the trajectory and features in 2D
	Input:
		pose:   4*4*N matrix representing the camera pose, 
				where N is the number of poses, and each
				4*4 matrix is in SE(3)
		m:		4*M matrix representing feature homogeneous coordinates 
				in world frame

landmarkMap:

	function to create landmark map for all time (features exceeding 50m in x or y filtered)
	Input:
		pose:   	4*4*N matrix representing the camera pose, 
					where N is the number of poses, and each
					4*4 matrix is in SE(3)
		features: 	4*M*N matrix representing selected features
					for all time in image frame
		M:			number of features
		T:			number of timesteps t
		K:			intrinsic camera calibration matrix 4*4
		b:			baseline - scalar
		imu_T_cam:	extrinsic transform from camera to imu 4*4 in SE(3)
		
	Output:
		m:			4*M matrix representing feature homogeneous coordinates 
					in world frame						
