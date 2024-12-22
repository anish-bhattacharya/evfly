#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import sys, select, termios, tty

# Function to read keyboard key presses
def getch():
    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # Set terminal to raw mode
        tty.setraw(sys.stdin.fileno())
        # Read a single character
        ch = sys.stdin.read(1)
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return ch

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('keyboard_publisher')

    # Create a publisher for the keyboard input
    pub = rospy.Publisher('keyboard_input', String, queue_size=10)
    
    # Set the publishing rate to 30Hz
    rate = rospy.Rate(30)

    # Main loop
    while not rospy.is_shutdown():
        # Read keyboard input
        key = getch()

        # Publish the key pressed to the topic
        pub.publish(key)

        # Sleep to maintain the desired publishing rate
        rate.sleep()
