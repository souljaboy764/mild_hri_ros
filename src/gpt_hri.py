#!/usr/bin/python

import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pepper_controller_server.srv import JointTarget
import argparse
import requests
import base64

from nuitrack_node import *

# parser = argparse.ArgumentParser(description='Nuitrack HR Testing')
# parser.add_argument('--action', type=str, required=True, metavar='ACTION', choices=['handshake', 'rocket'],
#                 help='Action to perform: handshake or rocket).')
# args = parser.parse_args()
rospy.init_node('pepper_reset_node')
rospy.wait_for_service('/pepper_dcm/RightArm_controller/goal')
send_target = rospy.ServiceProxy('/pepper_dcm/RightArm_controller/goal', JointTarget)
rate = rospy.Rate(100)
joint_trajectory = JointTrajectory()
joint_trajectory.header.frame_id = "base_footprint"
joint_trajectory.joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", 'RHand']
joint_trajectory.points.append(JointTrajectoryPoint())
joint_trajectory.points[0].effort = np.zeros(len(joint_trajectory.joint_names)).tolist()


username = "pepper"
password = "unk@LIT2022#"

def login(username, password):
    session = requests.Session()

    json_data = {
        'username': username,
        'password': password,
        'keepMeLoggedIn': True,
    }

    response = session.post('https://auth.litviva.com/api/firstfactor', json=json_data)
    if response.status_code == 200:
        return session
    else:
        raise Exception(f"Login failed with status code {response.status_code}: {response.text}")

session = login(username, password)

nuitrack = NuitrackROS(width=848, height=480, horizontal=False)
display_img, skeleton, stamp = nuitrack.update()
display_img, skeleton, stamp = nuitrack.update()
display_img, skeleton, stamp = nuitrack.update()
display_img, skeleton, stamp = nuitrack.update()
display_img, skeleton, stamp = nuitrack.update()

retval, buffer = cv2.imencode('.jpg', nuitrack._color_img)
jpg_as_text = base64.b64encode(buffer).decode('utf-8')
response = session.post(
        "https://api.litviva.com/gpt4v-greeter", 
        json={
            "base64": jpg_as_text,
        }
    ).json()

print(response)#[0]['name'])
for r in response:
    print(r)
    if r['name'] == 'shake_hands':

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

    elif r['name'] == 'fist_bump':
        joint_trajectory.points[0].effort[0] = 0.7
        joint_trajectory.points[0].positions = [0.4, -0.009, 0.6, 0.7, 1., 0.] # default standing angle values
        joint_trajectory.header.stamp = rospy.Time.now()
        send_target(joint_trajectory)
        # rospy.Rate(1).sleep()
        # joint_trajectory.points[0].effort[0] = 0.3
        # joint_trajectory.points[0].positions = [1.5708, -0.109, 0.7854, 0.009, 1.8239, 0.75] # default standing angle values
        # joint_trajectory.header.stamp = rospy.Time.now()
        # send_target(joint_trajectory)
        # rospy.Rate(10).sleep()
        # joint_trajectory.header.stamp = rospy.Time.now()
        # send_target(joint_trajectory)
        # rospy.Rate(10).sleep()
        # joint_trajectory.header.stamp = rospy.Time.now()
        # send_target(joint_trajectory)
        # rospy.Rate(10).sleep()
        rospy.signal_shutdown('done')
    # elif r['name'] == 'fist_bump':
    #     joint_trajectory.points[0].effort[0] = 0.7
    #     joint_trajectory.points[0].positions = [0., -1., 0.6, 0.7, 1., 0.] # default standing angle values
    #     joint_trajectory.header.stamp = rospy.Time.now()
    #     send_target(joint_trajectory)
    #     # rospy.Rate(1).sleep()
    #     # joint_trajectory.points[0].effort[0] = 0.3
    #     # joint_trajectory.points[0].positions = [1.5708, -0.109, 0.7854, 0.009, 1.8239, 0.75] # default standing angle values
    #     # joint_trajectory.header.stamp = rospy.Time.now()
    #     # send_target(joint_trajectory)
    #     # rospy.Rate(10).sleep()
    #     # joint_trajectory.header.stamp = rospy.Time.now()
    #     # send_target(joint_trajectory)
    #     # rospy.Rate(10).sleep()
    #     # joint_trajectory.header.stamp = rospy.Time.now()
    #     # send_target(joint_trajectory)
    #     # rospy.Rate(10).sleep()
    #     rospy.signal_shutdown('done')
