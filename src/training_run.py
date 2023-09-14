#!/usr/bin/python

import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pepper_controller_server.srv import JointTarget
import argparse

parser = argparse.ArgumentParser(description='Nuitrack HR Testing')
parser.add_argument('--action', type=str, required=True, metavar='ACTION', choices=['handshake', 'rocket'],
                help='Action to perform: handshake or rocket).')
args = parser.parse_args()
rospy.init_node('pepper_reset_node')
rospy.wait_for_service('/pepper_dcm/RightArm_controller/goal')
send_target = rospy.ServiceProxy('/pepper_dcm/RightArm_controller/goal', JointTarget)
rate = rospy.Rate(100)
joint_trajectory = JointTrajectory()
joint_trajectory.header.frame_id = "base_footprint"
joint_trajectory.joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", 'RHand']
joint_trajectory.points.append(JointTrajectoryPoint())
joint_trajectory.points[0].effort = np.zeros(len(joint_trajectory.joint_names)).tolist()

if args.action == 'handshake':
    joint_trajectory.points[0].effort[0] = 0.5
    joint_trajectory.points[0].positions = [0.4, -0.009, 0.6, 0.7, 1., 0.75] # default standing angle values
    joint_trajectory.header.stamp = rospy.Time.now()
    send_target(joint_trajectory)
    rospy.Rate(0.5).sleep()
    joint_trajectory.points[0].effort[0] = 0.2
    joint_trajectory.header.stamp = rospy.Time.now()
    send_target(joint_trajectory)
    rospy.Rate(1).sleep()
    rospy.signal_shutdown('done')

elif args.action == 'rocket':
    joint_trajectory.points[0].effort[0] = 0.7
    joint_trajectory.points[0].positions = [0.4, -0.009, 0.6, 0.7, 1., 0.] # default standing angle values
    joint_trajectory.header.stamp = rospy.Time.now()
    send_target(joint_trajectory)
    rospy.Rate(0.5).sleep()
    for j in np.linspace(0.4,-0.2,50):
        joint_trajectory.points[0].positions = [j, -0.009, 0.6, 0.7, 1., 0.] # default standing angle values
        joint_trajectory.header.stamp = rospy.Time.now()
        send_target(joint_trajectory)
        rospy.Rate(20).sleep()
    rospy.Rate(0.75).sleep()
    joint_trajectory.points[0].effort[0] = 0.3
    joint_trajectory.points[0].positions = [1.5708, -0.109, 0.7854, 0.009, 1.8239, 0.75] # default standing angle values
    joint_trajectory.header.stamp = rospy.Time.now()
    send_target(joint_trajectory)
    rospy.Rate(10).sleep()
    joint_trajectory.header.stamp = rospy.Time.now()
    send_target(joint_trajectory)
    rospy.Rate(10).sleep()
    joint_trajectory.header.stamp = rospy.Time.now()
    send_target(joint_trajectory)
    rospy.Rate(10).sleep()
    rospy.signal_shutdown('done')
