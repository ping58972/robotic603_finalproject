#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


def finite_ranges(values):
    return [value for value in values if math.isfinite(value) and value > 0.0]


class RealWallFollower:
    def __init__(self):
        rospy.init_node("real_wall_follower")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.desired_right_distance = float(rospy.get_param("~desired_right_distance", 0.55))
        self.front_stop_distance = float(rospy.get_param("~front_stop_distance", 0.65))
        self.linear_speed = float(rospy.get_param("~linear_speed", 0.18))
        self.turn_speed = float(rospy.get_param("~turn_speed", 0.55))
        self.wall_kp = float(rospy.get_param("~wall_kp", 1.1))
        self.search_turn_speed = float(rospy.get_param("~search_turn_speed", -0.25))
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.on_shutdown(self.stop)
        rospy.loginfo("real_wall_follower ready: scan=%s cmd_vel=%s", self.scan_topic, self.cmd_vel_topic)

    @staticmethod
    def sector_min(scan, start_deg, end_deg):
        start = RealWallFollower.normalize_angle(math.radians(start_deg))
        end = RealWallFollower.normalize_angle(math.radians(end_deg))
        sector_ranges = []
        for index, distance in enumerate(scan.ranges):
            angle = RealWallFollower.normalize_angle(scan.angle_min + index * scan.angle_increment)
            if RealWallFollower.angle_in_sector(angle, start, end):
                sector_ranges.append(distance)
        values = finite_ranges(sector_ranges)
        return min(values) if values else float("inf")

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def angle_in_sector(angle, start, end):
        if start <= end:
            return start <= angle <= end
        return angle >= start or angle <= end

    def scan_callback(self, scan):
        front = self.sector_min(scan, -20.0, 20.0)
        right = self.sector_min(scan, -105.0, -60.0)
        cmd = Twist()

        if front < self.front_stop_distance:
            cmd.linear.x = 0.0
            cmd.angular.z = self.turn_speed
        elif math.isfinite(right):
            error = self.desired_right_distance - right
            cmd.linear.x = self.linear_speed
            cmd.angular.z = max(-self.turn_speed, min(self.turn_speed, self.wall_kp * error))
        else:
            cmd.linear.x = self.linear_speed * 0.6
            cmd.angular.z = self.search_turn_speed

        self.cmd_pub.publish(cmd)

    def stop(self):
        self.cmd_pub.publish(Twist())


if __name__ == "__main__":
    RealWallFollower()
    rospy.spin()
