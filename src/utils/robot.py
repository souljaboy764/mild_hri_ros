import numpy as np
import scipy.optimize as opt
from check_selfcollision.srv import *
from moveit_msgs.msg import RobotState
from utils.helper import *
import rospy

# Pepper Joint limits
lower_bounds = np.array([-2.0857, -1.5621, -2.0857, 0.009])
upper_bounds = np.array([2.0857, -0.009, 2.0857, 1.5621])
bounds = ((lower_bounds[0], upper_bounds[0]),(lower_bounds[1], upper_bounds[1]),(lower_bounds[2], upper_bounds[2]),(lower_bounds[3], upper_bounds[3]))

# Assumes that the skeletons are already rotated such that the front 
def joint_angle_extraction(skeleton): # Based on the Pepper Robot Kinematics
	
	# T = np.eye(4)
	# T[:3,3] = skeleton[0]
	# T[:3,:3] = rotation_normalization(skeleton)
	# rightShoulder = skeleton[joints_idx["right_shoulder"]] - T[:3,3]
	# rightShoulder = T[:3,:3].dot(rightShoulder)
	# rightElbow = skeleton[joints_idx["right_elbow"]] - T[:3,3]
	# rightElbow = T[:3,:3].dot(rightElbow)
	# rightHand = skeleton[joints_idx["right_hand"]] - T[:3,3]
	# rightHand = T[:3,:3].dot(rightHand)

	# rightShoulder = skeleton[joints_idx["right_shoulder"]]
	# rightElbow = skeleton[joints_idx["right_elbow"]]
	# rightHand = skeleton[joints_idx["right_hand"]]

	rightShoulder = np.array([skeleton[-3,1], -skeleton[-3,0], skeleton[-3,2]])
	rightElbow = np.array([skeleton[-2,1], -skeleton[-2,0], skeleton[-2,2]])
	rightHand = np.array([skeleton[-1,1], -skeleton[-1,0], skeleton[-1,2]])
	
	# rightShoulder = skeleton[-3]
	# rightElbow = skeleton[-2]
	# rightHand = skeleton[-1]
	
	rightYaw = 0
	rightPitch = 0
	rightRoll = 0
	rightElbowAngle = 0
	
	# Recreating arm with upper and under arm
	rightUpperArm = rightElbow - rightShoulder
	rightUnderArm = rightHand - rightElbow

	rightElbowAngle = angle(rightUpperArm, rightUnderArm)

	armlengthRight = np.linalg.norm(rightUpperArm)
	rightYaw = np.arctan2(rightUpperArm[1],-rightUpperArm[2]) # Comes from robot structure
	# rightYaw -= 0.009
	rightPitch = np.arctan2(np.clip(rightUpperArm[0],0,armlengthRight), rightUpperArm[2]) # Comes from robot structure
	rightPitch -= np.pi/2

	# Recreating under Arm Position with known Angles(without roll)
	rightRotationAroundY = euler_matrix(0, rightPitch, 0,)[:3,:3]
	rightRotationAroundX = euler_matrix(0, 0, rightYaw)[:3,:3]
	rightElbowRotation = euler_matrix(0, 0, rightElbowAngle)[:3,:3]

	rightUnderArmInZeroPos = np.array([np.linalg.norm(rightUnderArm), 0, 0.])
	rightUnderArmWithoutRoll = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightElbowRotation,rightUnderArmInZeroPos)))

	# Calculating the angle betwenn actual under arm position and the one calculated without roll
	rightRoll = angle(rightUnderArmWithoutRoll, rightUnderArm)
	
	#This is a check which sign the angle has as the calculation only produces positive angles
	rightRotationAroundArm = euler_matrix(0, 0, -rightRoll)[:3, :3]
	rightShouldBeWristPos = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightRotationAroundArm,np.dot(rightElbowRotation,rightUnderArmInZeroPos))))
	r1saver = np.linalg.norm(rightUnderArm - rightShouldBeWristPos)
	
	rightRotationAroundArm = euler_matrix(0, 0, rightRoll)[:3, :3]
	rightShouldBeWristPos = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightRotationAroundArm,np.dot(rightElbowRotation,rightUnderArmInZeroPos))))
	r1 = np.linalg.norm(rightUnderArm - rightShouldBeWristPos)
	
	# if (r1 > r1saver):
	# 	rightRoll = -rightRoll

	return np.array([rightPitch, rightYaw, rightRoll, rightElbowAngle])

##
#   This class defines the class for implementing the forward kinematics of the pepper robot.
#   The code taken from the code by Sebastian Gomez-Gonzalez at https://github.com/sebasutp/promp
#
#   @author Vignesh Prasad <vignesh.prasad@tu-darmstadt.de>, TU Darmstadt
class PepperKinematics:

	##
	#	Initialization function for the class. When extending, add whatever parameters are necessary for implementing the `_link_matrices` function, such as end effector pose or arm lengths etc.
	#
	def __init__(self, lambda_x = 0.5, lambda_theta = 0.5):
		self.lambda_x = lambda_x
		self.lambda_theta = lambda_theta
		self._robot_state = RobotState()
		self._robot_state.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']
		rospy.wait_for_service('check_selfcollision_service')
		self._check_selfcollision_service = rospy.ServiceProxy('check_selfcollision_service', SelfCollosion)

	##
	#	Function implementing the relative linkwise forward kinematics of the robot. This is the main function that needs to be implemented while extending this base class.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return T list of N + 2 Transformation Matrices each  having dimensions 4 x 4. The first should be a transformation to the base link of the robot. The next N matrices are the result of applying each of the N degrees of freedom. The ith matrix is the relative transformation between the (i-1)th to the ith frame of reference  The last matrix if the transformation to the final end effector link from the last frame of reference.
	#
	# q = [RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll]
	def _link_matrices(self,q):

		m00 = euler_matrix(0.006, 0.000, 0.000) #base_link to right shoulder joint
		m00[2,3] = 0.82
		
		m01 = euler_matrix(0, q[0],0)
		m01[:3,3] = np.array([-0.057, -0.14974, 0.08682])

		m12 = euler_matrix(0, 0, q[1])
		
		m23 = euler_matrix(q[2], -0.157079, 0)
		m23[:3,3] = np.array([0.1812, -0.015, 0.00013])

		m34 = euler_matrix(0, 0, q[3])

		m45 = np.eye(4)
		m45[0,3] = 0.15
		return [m00,m01,m12,m23,m34,m45]


	##
	#	Function implementing the forward kinematics of the robot in the global frame of reference.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return absolute_transforms list of N + 1 Transformation Matrices each  having dimensions 4 x 4. The first N matrices represent the pose of the reference frames of each of the N degrees of freedon in the global frame. The ith matrix is the relative transformation between the global frame to the ith frame of reference  The last matrix if the transformation to the final end effector link.
	#
	def forward_kinematics(self,q):
		H = self._link_matrices(q)
		A = H[0]
		absolute_transforms = []
		for i in range(1,len(H)):
			A = np.dot(A,H[i])
			absolute_transforms.append(A)
		return absolute_transforms

	##
	#	Function to obtain the euler angles from a given rotation matrix.
	#
	#   @param rotMat The input rotation matrix.
	#
	#   @return eul Vector of size 3 with the yaw, pitch and roll angles respectively.
	#
	def __rotMatToEul(self,rotMat):
		eul = np.zeros(3)
		eul[0] = np.arctan2(-rotMat[2,1],rotMat[2,2])
		eul[1] = np.arctan2(rotMat[2,0],np.sqrt(rotMat[2,1]**2+rotMat[2,2]**2))
		eul[2] = np.arctan2(-rotMat[1,0],rotMat[0,0])
		return eul

	##
	#	Function to obtain 6DoF pose of the end effector.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return pos Vector of size 3 with the x,y,z positions of the end effector.
	#   @return orientation Vector of size 3 with the yaw, pitch and roll angles of the end effector.
	#
	def end_effector(self, q ,As=None):
		if As is None:
			As = self.forward_kinematics(q)
		end_eff = As[-1]
		pos = end_eff[0:3,3]
		orientation = self.__rotMatToEul(end_eff[0:3,0:3].transpose())
		return pos, orientation

	##
	#	Calculates the numerical jacobian of the forward kinemnatics of the robot.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#   @param eps Amount to perturb each degree of freedom with to calculate the jacobian.
	#
	#   @return jac Jacobian matrix of the x,y,z positions of the end effector having shape 3 x N.
	#
	def __num_jac(self, q, eps = 1e-6):
		jac = np.zeros((3,len(q)))
		fx,ori = self.end_effector(q)
		for i in range(len(q)):
			q[i] += eps
			fxh,ori = self.end_effector(q)
			jac[:,i] = (fxh - fx) / eps
			q[i] -= eps
		return jac

	##
	#	Calculates the analytical jacobian of the forward kinemnatics of the robot.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#   @param As list of N + 1 Transformation Matrices each  having dimensions 4 x 4. The first N matrices  represent the pose of the reference frames of each of the N degrees of freedon in the global frame. The ith matrix is the relative transformation between the global frame to the ith frame of reference  The last matrix if the transformation to the final end effector link.
	#
	#   @return jac Jacobian matrix of the 6DoF pose of the end effector having shape 6 x N.
	#
	def __analytic_jac(self, q, As=None):
		jac = np.zeros((6,len(q)))
		if As is None:
			As = self.forward_kinematics(q)
		pe = As[-1][0:3,3]
		for i in range(len(q)):
			zprev = As[i][0:3,2]
			pprev = As[i][0:3,3]
			jac[0:3,i] = cross(zprev, pe - pprev)
			jac[3:6,i] = zprev
		return jac

	##
	#	Calculates the jacobian of the forward kinemnatics of the robot.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return jac Jacobian matrix of the 6DoF pose of the end effector having shape 6 x N.
	#
	def jacobian(self, q):
		As = self.forward_kinematics(q)
		#jac = self.__num_jac(q,1e-6)
		jac = self.__analytic_jac(q, As)
		return jac

	##
	#	Calculates the position of the end effector and the jacobian of the forward kinemnatics of the robot.
	#
	#   @param q Vector of dimension N containing the values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return pos Vector of size 3 with the x,y,z positions of the end effector.
	#   @return jac Jacobian matrix of the 6DoF pose of the end effector having shape 6 x N.
	#   @return ori Vector of size 3 with the yaw, pitch and roll angles of the end effector.
	#
	def position_and_jac(self, q):
		As = self.forward_kinematics(q)
		jac = self.__analytic_jac(q, As)
		pos, ori = self.end_effector(q, As)
		return pos, jac, ori

	##
	#	Calculates the trajectory of the end effector given a set of joint configuration trajectories.
	#
	#   @param Q Matrix of dimension num_samples x N containing the trajectories of each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#
	#   @return pos Matrix of size num_samples x 3 containing the the x,y,z locations of the end effector.
	#   @return orientation Matrix of size num_samples x 3 with the yaw, pitch and roll angles of the end effector at each pointin the trajectory.
	#
	def end_eff_trajectory(self, Q):
		pos = []
		orientation = []
		for t in range(len(Q)):
			post, ort = self.end_effector(Q[t])
			pos.append(post)
			orientation.append(ort)
		return np.array(pos), np.array(orientation)

	##
	#	Calculates the cost function and gradient of the cost function for the Inverse Kinematics optimization.
	#
	#   @param theta Vector of dimension N containing the current estimate of values for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#   @param mu_theta Vector of dimension N containing the mean values of the prior for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#   @param inv_sigma_theta Matrix of dimension N x N containing the inverse of the covariance matrix of the prior for each degree of freedom of the robot.
	#   @param mu_x Vector of dimension 3 containing the mean values of the distribution for the goal's 3D coordinates in meters.
	#   @param inv_sigma_x Matrix of dimension 3 x 3 containing the inverse of the covariance matrix of the distribution for the goal's 3D coordinates.
	#
	#   @return nll Scalar value returning the current cost function value.
	#   @return grad_nll Vector of dimension N containing the gradient of the cost function w.r.t. the current estimate theta.
	#
	def __laplace_cost_and_grad(self, theta, mu_theta, inv_sigma_theta, mu_x, inv_sigma_x):
		f_th, jac_th, ori = self.position_and_jac(theta)
		jac_th = jac_th[0:3,:]
		diff1 = theta - mu_theta
		tmp1 = np.dot(inv_sigma_theta, diff1)
		diff2 = f_th - mu_x
		tmp2 = np.dot(inv_sigma_x, diff2)
		
		self._robot_state.joint_state.position = theta
		
		nll = 0.5*(np.dot(diff1,tmp1) + np.dot(diff2,tmp2)) + (not self._check_selfcollision_service(self._robot_state).is_colliding)*1000
		grad_nll = tmp1 + np.dot(jac_th.T,tmp2)

		return nll, grad_nll

	##
	#	Solves the Inverse Kinematics of the robot to reach a particular 3D location while trying to stick close to a prior distribution of the joint configuration.
	#
	#   @param mu_theta Vector of dimension N containing the mean values of the prior for each degree of freedom of the robot in radians (for angles) or meters (for translations).
	#   @param sig_theta Matrix of dimension N x N containing the covariance matrix of the prior for each degree of freedom of the robot.
	#   @param mu_x Vector of dimension 3 containing the mean values of the distribution for the goal's 3D coordinates in meters.
	#   @param sig_x Matrix of dimension 3 x 3 containing the covariance matrix of the distribution for the goal's 3D coordinates.
	#
	#   @param pos_mean Vector of dimension N containing the mean values of the posterior for each degree of freedom of the robot in radians (for angles) or meters (for translations) after solving the inverse kinematics.
	#   @param pos_cov Matrix of dimension N x N containing the covariance matrix of the posterior for each degree of freedom of the robot after solving the inverse kinematics.
	#
	def inv_kin(self, mu_theta, sig_theta, mu_x, sig_x, **kwargs):
		inv_sig_theta = np.linalg.inv(sig_theta)
		inv_sig_x = np.linalg.inv(sig_x)
		cost_grad = lambda theta: self.__laplace_cost_and_grad(theta, mu_theta, inv_sig_theta, mu_x, inv_sig_x)
		cost = lambda theta: cost_grad(theta)[0]
		grad = lambda theta: cost_grad(theta)[1]

		kwargs.setdefault('method', 'BFGS')
		kwargs.setdefault('jac', grad)
		res = opt.minimize(cost, mu_theta, **kwargs)
		kwargs.setdefault('print', False)
		post_mean = res.x
		if hasattr(res, 'hess_inv'):
			post_cov = res.hess_inv
		else:
			post_cov = None
		if kwargs['options']['disp']:
			print(res)
			print(not self._check_selfcollision_service(self._robot_state).is_colliding)
		return post_mean, post_cov
