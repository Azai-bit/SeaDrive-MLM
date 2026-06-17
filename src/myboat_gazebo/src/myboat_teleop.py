#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty

msg = """
Control myboat!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

c : increase linear speed by 0.25 (max 6.0 m/s)
d : decrease linear speed by 0.25 (min 0.0 m/s)
space key, k : force stop
anything else : stop smoothly

CTRL-C to quit
"""

moveBindings = {
    'i': (1, 0),
    'o': (1, -1),
    'j': (0, 1),
    'l': (0, -1),
    'u': (1, 1),
    ',': (-1, 0),
    '.': (-1, 1),
    'm': (-1, -1),
}

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

linear_speed = 1.0
ANGULAR_SPEED = 0.8

def vels(speed):
    return "currently:\tlinear speed %.2f m/s" % (speed)

if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('myboat_teleop')

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

    x = 0
    th = 0
    count = 0
    control_speed = 0.0
    control_turn = 0.0

    try:
        print(msg)
        print(vels(linear_speed))

        while True:
            key = getKey()

            if key in moveBindings.keys():
                x = moveBindings[key][0]
                th = moveBindings[key][1]
                count = 0

            elif key == 'c':
                linear_speed = min(6.0, linear_speed + 0.25)
                print(vels(linear_speed))

            elif key == 'd':
                linear_speed = max(0.0, linear_speed - 0.25)
                print(vels(linear_speed))

            elif key == ' ' or key == 'k':
                x = 0
                th = 0
                control_speed = 0
                control_turn = 0

            else:
                count += 1
                if count > 4:
                    x = 0
                    th = 0
                if key == '\x03':
                    break

            target_speed = linear_speed * x
            target_turn = ANGULAR_SPEED * th

            if target_speed > control_speed:
                control_speed = min(target_speed, control_speed + 0.05)
            elif target_speed < control_speed:
                control_speed = max(target_speed, control_speed - 0.2)
            else:
                control_speed = target_speed

            if target_turn > control_turn:
                control_turn = min(target_turn, control_turn + 0.05)
            elif target_turn < control_turn:
                control_turn = max(target_turn, control_turn - 0.05)
            else:
                control_turn = target_turn

            twist = Twist()
            twist.linear.x = control_speed
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = control_turn
            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)