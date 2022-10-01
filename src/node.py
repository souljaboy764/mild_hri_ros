rospy.init_node('mild_hri_moveit_node')

rate = rospy.Rate(10)

limits_max = [2.08567, -0.00872665, 2.08567, 1.56207]
limits_min = [-2.08567, -1.56207, -2.08567, 0.00872665]
bounds = ((limits_min[0], limits_max[0]),(limits_min[1], limits_max[1]),(limits_min[2], limits_max[2]),(limits_min[3], limits_max[3]))

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import *
import actionlib

# client = actionlib.SimpleActionClient('execute_trajectory', ExecuteTrajectoryAction)
# client.wait_for_server()

# goal = ExecuteTrajectoryGoal()

# goal.trajectory.joint_trajectory.header.frame_id = "base_link"
# # joint_trajectory.joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
# goal.trajectory.joint_trajectory.joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
# goal.trajectory.joint_trajectory.points.append(JointTrajectoryPoint())
# goal.trajectory.joint_trajectory.points[0].time_from_start = rospy.Duration.from_sec(0.02)

robot_traj_publisher = rospy.Publisher("/pepper_dcm/RightArm_controller/command", JointTrajectory, queue_size=1)
joint_trajectory = JointTrajectory()
joint_trajectory.header.frame_id = "base_link"
joint_trajectory.joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
# joint_trajectory.joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
joint_trajectory.points.append(JointTrajectoryPoint())
joint_trajectory.points[0].time_from_start = rospy.Duration.from_sec(0.05)

robotstate_pub = rospy.Publisher("display_robot_state", DisplayRobotState, queue_size=5)
robotmsg = DisplayRobotState()
# robotmsg.state.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
robotmsg.state.joint_state.name = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
print(robotmsg.state.joint_state.name)
# rospy.loginfo("CREATED MOVEGROUP")
left_arm_joints = np.deg2rad(np.array([90,5,-90,-1,0]))
# arm_joints = np.array([ 1.57079633, 0.08726646, -1.57079633, -0.01745329, 0., 1.28193268, -0.2122131, 1.12623912, 1.07140932, 0.])



predictor = MILDHRIROS()

while predictor.robot_command is None and not rospy.is_shutdown():
	rate.sleep()

if rospy.is_shutdown():
	exit(-1)

while not rospy.is_shutdown():
	rate.sleep()
	last_q = np.clip(predictor.robot_command, limits_min, limits_max)
	right_arm_joints = np.hstack([last_q, [0.]])

	# goal.trajectory.joint_trajectory.points[0].positions = np.concatenate([left_arm_joints, right_arm_joints])
	# print(goal.trajectory.joint_trajectory.points[0].positions)
	# goal.trajectory.joint_trajectory.header.stamp = rospy.Time.now()
	# client.send_goal(goal)

	joint_trajectory.points[0].positions = right_arm_joints
	joint_trajectory.header.stamp = rospy.Time.now()
	robot_traj_publisher.publish(joint_trajectory)
	rate.sleep()
	robot_traj_publisher.publish(joint_trajectory)
	rate.sleep()
	robot_traj_publisher.publish(joint_trajectory)
	rate.sleep()

	robotmsg.state.joint_state.position = np.concatenate([left_arm_joints, right_arm_joints])
	# print(last_q, robotmsg.state.joint_state.position)
	robotstate_pub.publish(robotmsg)

