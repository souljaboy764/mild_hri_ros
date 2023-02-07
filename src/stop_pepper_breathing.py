import qi
session = qi.Session()
session.connect("tcp://192.168.100.122:9559")
motion_service  = session.service("ALMotion")
if not motion_service.robotIsWakeUp():
	motion_service.wakeUp()
motion_service.setBreathEnabled('Body', False)
session.close()