#!/usr/bin/python

import rospy,sys
from std_msgs.msg import String
import markovify
from markovify import *
textFile = sys.argv[1]
# using markovify to create markov chain from text
def generateMarkovifyChain(text):
    text = open(text).read()
    markovModel = markovify.Text(text,
                                 retain_original = False)
    return markovModel.make_sentence()
# publishing randomly generated sentences from topic "markovTalker" at 
# frequency of 10 hz
def publishMarkovChain():
    pmarkov = rospy.Publisher("markovTalker",String,queue_size=10)
    rospy.init_node("markovPublish",anonymous=True)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # rospy.loginfo(chain) publishes to the console
        sentence = generateMarkovifyChain(textFile) 
        pmarkov.publish(sentence) # publishes each sentence from textfile in
        # first system argument
        rate.sleep()

if __name__ == "__main__":
    try:
        publishMarkovChain()
    except rospy.ROSInterruptException:
        pass
        
