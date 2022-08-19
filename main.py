from re import I
import numpy as np
from pr3_utils import *
import matplotlib.pyplot as plt
from scipy.linalg import expm

# FOR TESTING CODE
# exec(open("./main.py").read()) # use for testing code in python terminal

if __name__ == '__main__':

	# Load the measurements
	filename = "./data/03.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

	# Reduce Computation Time and Memory by Deleting Features
	for _ in range(3):
		features = np.delete(features, slice(None, None, 2),axis=1)

	M = len(features[0,:,0]) 	# number of features/landmarks
	T = len(t[0])-1  			# time stamps
	W = np.zeros((6,6))			# motion noise
	[W[0,0],W[1,1],W[2,2],W[3,3],W[4,4],W[5,5]] = [0.3,0.3,0.3,0.05,0.05,0.05]
	v = 0.3		# observation noise multiple

	features = np.delete(features, 0, axis=2) # delete first time stamp of features	

	#########################################
	# (a) IMU Localization via EKF Prediction
	
	mu_imu, Sigma_imu = prediction(t,linear_velocity,angular_velocity,W)	# mean and covariance of imu pose
		
	# (b) Landmark Mapping via EKF Update
	
	m = landmarkMap(mu_imu,features,M,T,K,b,imu_T_cam)		# coordinates of features wrt mean imu trajectory
	visualize_trajectory_2d(mu_imu,m,path_name = "Trajectory",show_ori=True)
	
	mu_map, Sigma_map = update(features,M,T,K,b,mu_imu,imu_T_cam,v)
	visualize_trajectory_2d(mu_imu,mu_map,path_name = "Trajectory",show_ori=True)
	
	# (c) Visual-Inertial SLAM

	# Not functional
	#mu_pose, mu_map, Sigma_joint = viSLAM(features,M,T,K,b,mu_imu,imu_T_cam,v,t,linear_velocity,angular_velocity,W)