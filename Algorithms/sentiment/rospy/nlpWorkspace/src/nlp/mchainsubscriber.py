#!/usr/bin/python 
# mchainsubscriber.py
import rospy
from std_msgs.msg import *

def callback(data):
    print("Comment:",data.data)
    
def markovSubscriber():
    rospy.init_node("markovSubscriber",anonymous=True)
    rospy.Subscriber("markovTalker",String,callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        markovSubscriber()
    except:
        pass
