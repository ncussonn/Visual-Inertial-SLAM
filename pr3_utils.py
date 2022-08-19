import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from scipy.linalg import expm

def load_data(file_name):
	'''
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
	'''
	with np.load(file_name) as data:
	
		t = data["time_stamps"] # time_stamps
		features = data["features"] # 4 x num_features : pixel coordinates of features
		linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
		angular_velocity = data["angular_velocity"] # angular velocity measured in the body frame
		K = data["K"] # intrinsic calibration matrix
		b = data["b"] # baseline
		imu_T_cam = data["imu_T_cam"] # Transformation from left camera to imu frame 
	
	return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam

def hatMapSetup(T,V,w):
	'''
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
		u_curly_hat:	6*6*t matrix for u curly hat
		v_hat:			3*3*t matrix for v hat
		w_hat:			3*3*t matrix for w hat
	'''
	
	v = np.delete(V,0,1)	# linear velocity
	w = np.delete(w,0,1)	# angular velocity
		
	# Vectorizing w_hat and u_hat
	w_hat = np.zeros((3,3,T))

	w_hat[0,0,:] = 0
	w_hat[0,1,:] = -w[2]
	w_hat[0,2,:] = w[1]
	w_hat[1,0,:] = w[2]
	w_hat[1,1,:] = 0
	w_hat[1,2,:] = -w[0]
	w_hat[2,0,:] = -w[1]
	w_hat[2,1,:] = w[0]
	w_hat[2,2,:] = 0

	v_hat = np.zeros((3,3,T))

	v_hat[0,0,:] = 0
	v_hat[0,1,:] = -v[2]
	v_hat[0,2,:] = v[1]
	v_hat[1,0,:] = v[2]
	v_hat[1,1,:] = 0
	v_hat[1,2,:] = -v[0]
	v_hat[2,0,:] = -v[1]
	v_hat[2,1,:] = v[0]
	v_hat[2,2,:] = 0

	u_hat = np.zeros((4,4,T))

	u_hat[0,0,:] = w_hat[0,0]
	u_hat[0,1,:] = w_hat[0,1]
	u_hat[0,2,:] = w_hat[0,2]
	u_hat[0,3,:] = v[0]
	u_hat[1,0,:] = w_hat[1,0]
	u_hat[1,1,:] = w_hat[1,1]
	u_hat[1,2,:] = w_hat[1,2]
	u_hat[1,3,:] = v[1]
	u_hat[2,0,:] = w_hat[0,0]
	u_hat[2,1,:] = w_hat[0,0]
	u_hat[2,2,:] = w_hat[0,0]
	u_hat[2,3,:] = v[2]
	u_hat[3,0,:] = 0
	u_hat[3,1,:] = 0
	u_hat[3,2,:] = 0
	u_hat[3,3,:] = 0

	u_curly_hat = np.zeros((6,6,T))

	for i in range(0,T):
		u_curly_hat[:,:,i] = np.block([[w_hat[:,:,i],v_hat[:,:,i]],[np.zeros((3,3)),w_hat[:,:,i]]])

	return u_hat, u_curly_hat, v_hat, w_hat

def prediction(t,V,w,W):
	'''
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
	'''
	T = len(t[0])-1			# time step count
	tau = np.diff(t)		# discrete time
	
	shape = (6,6,T)
	Sigma = np.zeros(shape)
	idx = np.arange(shape[0])
	Sigma[idx, idx, :] = 0.05	# covariance for each time step

	mu = np.zeros((4,4,T)) 
	mu[:,:,0] = np.eye(4) 	# Initial pose in line with origin
	mu[1,1,0] = -1			# IMU is inverted
	mu[2,2,0] = -1			# IMU is inverted
	
	# Vectorizing hats
	u_hat, u_curly_hat, v_hat, w_hat = hatMapSetup(T,V,w)

	tau_times_u_hat = tau*u_hat
	tau_curlyHat = tau*u_curly_hat

	for i in range(0,T-1):
		
		mu[:,:,i+1] = mu[:,:,i] @ expm(tau_times_u_hat[:,:,i])	# mean nominal Trajectory
		Sigma[:,:,i+1] = expm(-tau_curlyHat[:,:,i]) @ Sigma[:,:,i] @ np.transpose(expm(-tau_curlyHat[:,:,i])) + W	# covariance of IMU pose

	return mu, Sigma

def update(features,M,T,K,b,pose,imu_T_cam,v):
	'''
	function to do Visual Mapping via the EKF update
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
		
	Output:
		mu:			4*M matrix representing final mean map feature coordinates in world frame
		Sigma:		3M*3M matrix representing observation covariances
	'''
	# Initializing Parameters
	[fsu,fsv,cu,cv]=[K[0,0],K[1,1],K[0,2],K[1,2]]	# intrinsic Parameters
	Ks = np.array([[fsu,0,cu,0],[0,fsv,cv,0],[fsu,0,cu,-fsu*b],[0,fsv,cv,0]])
	mu = np.zeros((3*M))				# mean landmark map	(world coords)
	Sigma = (np.eye((3*M)))*0.05	# covariance matrix
	P = np.block([np.eye(3),np.zeros((3,1))])	# projection matrix
	I = np.eye(3*M)
	
	for t in range(0,T): # loop thru time

		# Redefine Variables dependent on timestep
		[ul,vl,ur,vr] = features[:,:,t]		# pixel coordinates of features at timestep
		N = np.size(np.where(ul != -1))		# number of observed features at timestep
		Z = np.zeros((4*N))  				# initialize feature coordinates in camera frame (pixels)
		Z_tilde = np.zeros((4*N)) 			# initialize predicted feature coordinates	(pixels)
		H = np.zeros((4*N,3*M))				# observation model Jacobian for current timestep
		KalmanGain = np.zeros((4*N,3*M))	# Kalman Gain for current timestep
	
		for j in range(0,M):	# loop thru features at current time

			if ul[j] != -1:		# if feature is observed, update using it

				for i in range(0,N):	# loop thru observed features at current time
					
					Z[4*i:4*i+4] = features[:,j,t] # observed feature pixel coordinates

					# REIMPLEMENT IF TIME ALLOWS
					# Only use features that are close in x,y
					#if abs(x) < 50 and abs(y) < 50:	# filter points more than 50 meters away in x or y
										
					if t == 0: # CREATE PRIOR MEAN
						
						d = ul[j]-ur[j]	# disparity
						# R3 Pixel Coordinates in optical frame
						z = fsu*b/d
						[x,y] = [z*(ul[j]-cu)/fsu, z*(vl[j]-cv)/fsv]
						z_init = np.ones((4,))
						z_init[0:3] = [x,y,z]
						mu[3*j:3*j+3] = np.delete((pose[:,:,t] @ imu_T_cam @ z_init),3)	# update feature on mean map using initial observation
												
					else: # Update as normal
						q =  np.linalg.inv(imu_T_cam) @ np.linalg.inv(pose[:,:,t]) @ np.concatenate((mu[3*j:3*j+3],[1]))	# observed point (inside of projection fnct pi)
						# Pi function
						observedPoint = np.ones((4,))
						observedPoint[0] = q[0]/abs(q[2])
						observedPoint[1] = q[1]/abs(q[2])
						observedPoint[2] = q[2]/abs(q[2])
						observedPoint[3] = q[3]/abs(q[2]) 	# divide each element by depth (yes I know there is a better way to do this, but time)
						pi = observedPoint					# projection function
						Z_tilde[4*i:4*i+4] = Ks @ pi		# predicted observation using prior mean feature location in world frame															
																	
						# UPDATE STEP
						# 1.) Compute Observation Model Jacobian
						dpi_dq = 1/q[2]*np.array([[1,0,-q[0]/q[2],0],[0,1,-q[1]/q[2],0],[0,0,0,0],[0,0,-q[3]/q[2],1]])
						H[4*i:4*i+4,3*j:3*j+3] = -Ks @ dpi_dq @ np.linalg.inv(imu_T_cam) @ np.linalg.inv(pose[:,:,t]) @ np.transpose(P)	# Jacobian of observation model for feature j at time t

		if t != 0: # CONTINUE UPDATE STEP

			# 2.) Compute Kalman Gain
			IxV = np.eye((4*N))*v		# Observation Noise matrix I xo V
			KalmanGain = Sigma @ np.transpose(H) @ np.linalg.inv(H @ Sigma @ np.transpose(H) + IxV)

			# 3.) Compute Updated Mean
			mu = mu + KalmanGain @ (Z-Z_tilde)	# feature means at current time
				
			# 4.) Compute Updated Covariance
			Sigma = (I - KalmanGain @ H) @ Sigma
					
	# Get mu into homogeneous coordinates and right shape for visualization function
	mu = np.reshape(mu,(M,3))
	ones = np.ones((M,1))
	mu = np.concatenate([mu,ones],axis=1).T
	
	return mu, Sigma

'''
def viSLAM(features,M,T,K,b,imu_T_cam,v,time_stamp,V,w,W):
	
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
				with shape 4*M
		Sigma: covariance of Sigma for IMU pose and each feature
			with shape 3M+6*3M+6
	
	mu_pose = np.zeros((4,4,T)) 
	mu_pose[:,:,0] = np.eye(4) 				# Initial pose in line with origin
	mu_pose[1,1,0] = -1						# IMU is inverted
	mu_pose[2,2,0] = -1						# IMU is inverted
	Sigma_pose = np.eye(6)*0.05				# pose covariance
	Sigma_cross = np.zeros((3*M,6))*0.05	# cross covariance
	Sigma_map = (np.eye((3*M)))*0.05		# map covariance
	Sigma = np.block([[Sigma_map,Sigma_cross],[Sigma_cross.T,Sigma_pose]])	# joint corariance 3*M+6 x 3*M+6
	P = np.block([np.eye(3),np.zeros((3,1))])	# projection matrix
	I = np.eye(3*M)
	tau = np.diff(time_stamp)			# discrete time
	[fsu,fsv,cu,cv]=[K[0,0],K[1,1],K[0,2],K[1,2]]	# intrinsic Parameters
	Ks = np.array([[fsu,0,cu,0],[0,fsv,cv,0],[fsu,0,cu,-fsu*b],[0,fsv,cv,0]])

	# Vectorizing hats before computation
	u_hat, u_curly_hat, _, _ = hatMapSetup(T,V,w)
	tau_times_u_hat = tau*u_hat
	tau_curlyHat = tau*u_curly_hat	

	for t in range(0,T):

		#PREDICT POSE AT CURRENT TIME#		
		mu_pose[:,:,t+1] = mu_pose[:,:,t] @ expm(tau_times_u_hat[:,:,t])	# mean nominal Trajectory prediction
		Sigma[-6:-1,-6:-1] = expm(-tau_curlyHat[:,:,t]) @ Sigma[-6:-1,-6:-1] @ np.transpose(expm(-tau_curlyHat[:,:,t])) + W	# covariance of IMU pose

		#UPDATE FEATURES USING PRIOR POSE#

		# Redefine Variables dependent on timestep
		[ul,vl,ur,vr] = features[:,:,t]		# pixel coordinates of features at timestep
		N = np.size(np.where(ul != -1))		# number of observed features at timestep
		Z = np.zeros((4*N))  				# initialize feature coordinates in camera frame (pixels)
		Z_tilde = np.zeros((4*N)) 			# initialize predicted feature coordinates	(pixels)
		H = np.zeros((4*N,3*M+6))			# observation model Jacobian for current timestep
		KalmanGain = np.zeros((3*M+6,4*N))	# Kalman Gain for current timestep
	
		for j in range(0,M):	# loop thru features at current time

			if ul[j] != -1:		# if feature is observed, update using it

				for i in range(0,N):	# loop thru observed features at current time
					
					Z[4*i:4*i+4] = features[:,j,t] # observed feature pixel coordinates

					# REIMPLEMENT IF TIME ALLOWS
					# Only use features that are close in x,y
					#if abs(x) < 50 and abs(y) < 50:	# filter points more than 50 meters away in x or y
										
					if t == 0: # CREATE PRIOR MEAN
						
						d = ul[j]-ur[j]	# disparity
						# R3 Pixel Coordinates in optical frame
						z = fsu*b/d
						[x,y] = [z*(ul[j]-cu)/fsu, z*(vl[j]-cv)/fsv]
						z_init = np.ones((4,))
						z_init[0:3] = [x,y,z]
						mu[3*j:3*j+3] = np.delete((mu_pose[:,:,t] @ imu_T_cam @ z_init),3)	# update feature on mean map using initial observation
												
					else: # Update as normal
						q =  np.linalg.inv(imu_T_cam) @ np.linalg.inv(mu_pose[:,:,t]) @ np.concatenate((mu[3*j:3*j+3],[1]))	# observed point (inside of projection fnct pi)
						# Pi function
						observedPoint = np.ones((4,))
						observedPoint[0] = q[0]/abs(q[2])
						observedPoint[1] = q[1]/abs(q[2])
						observedPoint[2] = q[2]/abs(q[2])
						observedPoint[3] = q[3]/abs(q[2]) 	# divide each element by depth (yes I know there is a better way to do this, but time)
						pi = observedPoint					# projection function
						Z_tilde[4*i:4*i+4] = Ks @ pi		# predicted observation using prior mean feature location in world frame															
																	
						# UPDATE STEP
						# 1.) Compute Observation Model Jacobian
						dpi_dq = 1/q[2]*np.array([[1,0,-q[0]/q[2],0],[0,1,-q[1]/q[2],0],[0,0,0,0],[0,0,-q[3]/q[2],1]])
						H[4*i:4*i+4,3*j:3*j+3] = -Ks @ dpi_dq @ np.linalg.inv(imu_T_cam) @ np.linalg.inv(pose[:,:,t]) @ np.transpose(P)	# Jacobian of observation model for feature j at time t

			# UPDATE POSE USING PRIOR FEATURES#

			# pseudocode
			H[pose_slice] = -Ks @ dpi/dq @ (np.linalg.inv(imu_T_cam) @ np.linalg.inv(mu_pose) @ mu_map) @ np.linalg.inv(imu_T_cam) circledotoperator # figure out what this is

			H = concatenated H_pose and H_map

			z_tilde = Ks @ pi(np.linalg.inv(imu_T_cam) @ np.linalg.inv(mu_pose) @ mu_map[j])

			mu_pose = mu_pose @ expm(hat(K(Z-Z_tilde)))	# UPDATE POSE

		if t != 0: # CONTINUE UPDATE STEP

			# 2.) Compute Kalman Gain
			IxV = np.eye((4*N))*v		# Observation Noise matrix I xo V
			KalmanGain = Sigma @ np.transpose(H) @ np.linalg.inv(H @ Sigma @ np.transpose(H) + IxV)

			# 3.) Compute Updated Mean
			mu_map = mu_map + KalmanGain @ (Z-Z_tilde)	# UPDATE MU OF MAP
				
			# 4.) Compute Updated Covariance
			Sigma = (I - KalmanGain @ H) @ Sigma	# UPDATE JOINT COVARIANCE
					
	# Get mu into homogeneous coordinates and right shape for visualization function
	mu_map = np.reshape(mu_map,(M,3))
	ones = np.ones((M,1))
	mu_map = np.concatenate([mu_map,ones],axis=1).T

	return mu_map, Sigma
'''

# MODIFIED TO PLOT FEATURES
def visualize_trajectory_2d(pose,m,path_name="Unknown",show_ori=False):
	'''
	function to visualize the trajectory and features in 2D
	Input:
		pose:   4*4*N matrix representing the camera pose, 
				where N is the number of poses, and each
				4*4 matrix is in SE(3)
		m:		4*M matrix representing feature homogeneous coordinates 
				in world frame
	'''
	fig,ax = plt.subplots(figsize=(5,5))
	n_pose = pose.shape[2]
	ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)	#plot trajectory
	ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
	ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
	ax.scatter(m[0,:],m[1,:],c = 'green',marker='.')		#plot features
	  
	if show_ori:
		select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
		yaw_list = []
		
		for i in select_ori_index:
			_,_,yaw = mat2euler(pose[:3,:3,i])
			yaw_list.append(yaw)
	
		dx = np.cos(yaw_list)
		dy = np.sin(yaw_list)
		dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
		ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
			color="b",units="xy",width=1)
	
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.axis('equal')
	ax.grid(False)
	ax.legend()
	plt.show(block=True)

	return fig, ax

def landmarkMap(pose,features,M,T,K,b,imu_T_cam):

	'''
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
	'''
	# Initializing Parameters
	[fsu,fsv,cu,cv]=[K[0,0],K[1,1],K[0,2],K[1,2]]	# Intrinsic Parameters
	[ul,vl,ur,vr] = features 						# pixel coordinates of all features M*N (M is feature number, N is timestamp)
						
	m = np.zeros((4,M))	# initialize landmark map 4*M
	s = np.ones((4,1)) # initialize feature coordinates in camera frame
	
	for t in range(0,T):	# loop thru time
		# Observations
		[ul,vl,ur,vr] = features[:,:,t]		# pixel coordinates of features at timestep
		
		for j in range(0,M):	# loop thru features at current time
			if ul[j] != -1:		# if feature observed, update it
				d = ul[j]-ur[j]	# disparity
				# R3 Pixel Coordinates in optical frame
				z = fsu*b/d
				x = z*(ul[j]-cu)/fsu
				y = z*(vl[j]-cv)/fsv

				# Only use features that are close in x,y
				if abs(x) < 50 and abs(y) < 50:	# filter points more than 50 meters away in x or y
					# Feature cam coordinates in homogeneous form
					s[0] = x
					s[1] = y
					s[2] = z
					# Position of feature in world frame (IMU is origin)
					# imu_T_cam integrates oRr matrix
					sbar = pose[:,:,t] @ imu_T_cam @ s
					m[:,j] = np.reshape(sbar,(4,))	# Update map

	return m
