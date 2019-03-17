#!/usr/bin/python 
# mchainsubscriber.py
import rospy
from std_msgs.msg import *
# displays subscriber data to console as "Comment: data.data" where data.data is comment 
def callback(data):
    print("Comment:",data.data)
# subscribes to topic of "markovTalker"
def markovSubscriber():
    rospy.init_node("markovSubscriber",anonymous=True)
    rospy.Subscriber("markovTalker",String,callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        markovSubscriber()
    except:
        pass
