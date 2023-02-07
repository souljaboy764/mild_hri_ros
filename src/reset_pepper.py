import qi
session = qi.Session()
session.connect("tcp://192.168.100.122:9559")
motion_service  = session.service("ALMotion")
if not motion_service.robotIsWakeUp():
	motion_service.wakeUp()
motion_service.setBreathEnabled('Body', False)
arm_joints = [1.57079633, -0.08726646, 1.57079633, 0.01745329] # default standing angle values
joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']
motion_service.setAngles(joint_names, arm_joints, 0.5)
session.close()
