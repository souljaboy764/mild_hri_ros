import numpy as np
from geometry_msgs.msg import Quaternion, Transform, Vector3, Pose
from utils.transformations import *
from utils.nuitrack import joints_idx

def angle(a,b):
	dot = np.dot(a,b)
	return np.arccos(dot/(np.linalg.norm(a)*np.linalg.norm(b)))

def projectToPlane(plane, vec):
	return (vec - plane)*np.dot(plane,vec)

def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
	return np.cross(a,b)

def rotation_normalization(skeleton):
	leftShoulder = skeleton[joints_idx["left_shoulder"]-1]
	rightShoulder = skeleton[joints_idx["right_shoulder"]-1]
	waist = skeleton[joints_idx["waist"]-1]
	
	yAxisHelper = waist - rightShoulder
	xAxis = leftShoulder - rightShoulder # right to left
	yAxis = cross(xAxis, yAxisHelper) # out of the human(like an arrow in the back)
	zAxis = cross(xAxis, yAxis) # like spine, but straight
	
	xAxis /= np.linalg.norm(xAxis)
	yAxis /= np.linalg.norm(yAxis)
	zAxis /= np.linalg.norm(zAxis)

	return np.array([[xAxis[0], xAxis[1], xAxis[2]],
					 [yAxis[0], yAxis[1], yAxis[2]],
					 [zAxis[0], zAxis[1], zAxis[2]]])


def mat2ROS(T:np.ndarray)->tuple:
	if T.shape == (4,4):
		return Vector3(*T[:3,3].tolist()), Quaternion(*quaternion_from_matrix(T))
	elif T.shape == (3,):
		return Vector3(*T.tolist()), Quaternion(0,0,0,1)

def ROS2mat(msg :(Transform or Pose))->np.ndarray:
	if isinstance(msg, Transform):
		return TF2mat(msg)
	elif isinstance(msg, Pose):
		return Pose2mat(msg)

def mat2TF(T:np.ndarray)->Transform:
	assert(T.shape == (4,4) or T.shape == (3,))
	return Transform(*mat2ROS(T))

def mat2Pose(T:np.ndarray)->Transform:
	assert(T.shape == (4,4) or T.shape == (3,))
	return Pose(*mat2ROS(T))

def TF2vec(transform:Transform)->np.ndarray:
	return np.array([transform.translation.x, transform.translation.y, transform.translation.z])

def TF2mat(transform:Transform)->np.ndarray:
	T = quaternion_matrix([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
	T[:3,3] = TF2vec(transform)
	return T

def Pose2vec(pose:Pose)->np.ndarray:
	return np.array([pose.position.x, pose.position.y, pose.position.z])

def Pose2mat(pose:Pose)->np.ndarray:
	T = quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
	T[:3,3] = Pose2vec(pose)
	return T