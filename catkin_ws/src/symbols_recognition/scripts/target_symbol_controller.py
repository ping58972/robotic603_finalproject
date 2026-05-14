#!/usr/bin/env python3
import json
import math
import sys

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


VALID_SYMBOLS = ("star", "square", "circle", "triangle")
SYMBOL_ALIASES = {
    "stars": "star",
    "squares": "square",
    "circles": "circle",
    "triangles": "triangle",
    "triange": "triangle",
    "trianlge": "triangle",
}


def normalize_symbol(value):
    symbol = str(value).strip().lower()
    symbol = SYMBOL_ALIASES.get(symbol, symbol)
    if symbol not in VALID_SYMBOLS:
        raise ValueError("Invalid target symbol '{}'. Choose one of: {}".format(value, ", ".join(VALID_SYMBOLS)))
    return symbol


def prompt_for_symbol():
    prompt = "Target symbol [star, square, circle, triangle]: "
    while not rospy.is_shutdown():
        try:
            value = input(prompt)
        except EOFError as exc:
            raise ValueError("Could not read target symbol from stdin. Pass target_symbol:=star.") from exc
        try:
            return normalize_symbol(value)
        except ValueError as exc:
            print(exc, file=sys.stderr)
    raise rospy.ROSInterruptException()


def finite_ranges(values):
    return [value for value in values if math.isfinite(value) and value > 0.0]


class TargetSymbolController:
    def __init__(self):
        rospy.init_node("target_symbol_controller")

        raw_target = rospy.get_param("~target_symbol", "").strip()
        if raw_target.lower() in ("", "prompt", "ask"):
            if not sys.stdin.isatty():
                raise ValueError("target_symbol was not provided and stdin is not interactive. Pass target_symbol:=star.")
            self.target_symbol = prompt_for_symbol()
        else:
            self.target_symbol = normalize_symbol(raw_target)

        self.result_topic = rospy.get_param("~result_topic", "/symbol_classifier/result")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.min_confidence = float(rospy.get_param("~min_confidence", 0.75))
        self.detection_timeout = float(rospy.get_param("~detection_timeout", 0.8))
        self.stop_distance = float(rospy.get_param("~stop_distance", 0.65))
        self.front_sector_degrees = float(rospy.get_param("~front_sector_degrees", 24.0))
        self.search_angular_speed = float(rospy.get_param("~search_angular_speed", 0.32))
        self.max_angular_speed = float(rospy.get_param("~max_angular_speed", 0.55))
        self.align_kp = float(rospy.get_param("~align_kp", 1.6))
        self.approach_speed = float(rospy.get_param("~approach_speed", 0.16))
        self.slow_approach_speed = float(rospy.get_param("~slow_approach_speed", 0.08))
        self.center_deadband = float(rospy.get_param("~center_deadband", 0.06))
        self.enable_motion = self.ros_bool(rospy.get_param("~enable_motion", True))
        self.require_bbox = self.ros_bool(rospy.get_param("~require_bbox", True))
        self.require_classifier_ready = self.ros_bool(rospy.get_param("~require_classifier_ready", True))
        self.classifier_timeout = float(rospy.get_param("~classifier_timeout", 2.0))

        self.front_distance = float("inf")
        self.last_target_detection = None
        self.last_classifier_result_time = None
        self.stopped = False

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.result_sub = rospy.Subscriber(self.result_topic, String, self.result_callback, queue_size=10)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.on_shutdown(self.stop_robot)

        rospy.loginfo(
            "target_symbol_controller ready: target=%s result=%s scan=%s cmd_vel=%s",
            self.target_symbol,
            self.result_topic,
            self.scan_topic,
            self.cmd_vel_topic,
        )

    @staticmethod
    def ros_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def angle_in_sector(angle, start, end):
        if start <= end:
            return start <= angle <= end
        return angle >= start or angle <= end

    def sector_min(self, scan, start_deg, end_deg):
        start = self.normalize_angle(math.radians(start_deg))
        end = self.normalize_angle(math.radians(end_deg))
        values = []
        for index, distance in enumerate(scan.ranges):
            angle = self.normalize_angle(scan.angle_min + index * scan.angle_increment)
            if self.angle_in_sector(angle, start, end):
                values.append(distance)
        finite = finite_ranges(values)
        return min(finite) if finite else float("inf")

    def scan_callback(self, scan):
        half_width = self.front_sector_degrees / 2.0
        self.front_distance = self.sector_min(scan, -half_width, half_width)

    def result_callback(self, msg):
        try:
            result = json.loads(msg.data)
        except ValueError:
            rospy.logwarn_throttle(2.0, "Ignoring malformed classifier result: %s", msg.data)
            return
        self.last_classifier_result_time = rospy.Time.now()

        try:
            raw_label = normalize_symbol(result.get("raw_label", result.get("label", "")))
        except ValueError:
            return
        accepted_label = result.get("label", "")
        confidence = float(result.get("confidence", 0.0))
        if raw_label != self.target_symbol or accepted_label == "unknown" or confidence < self.min_confidence:
            return
        if self.require_bbox and not result.get("bbox"):
            return

        self.last_target_detection = {
            "time": rospy.Time.now(),
            "center_x": float(result.get("center_x", 0.5)),
            "center_y": float(result.get("center_y", 0.5)),
            "confidence": confidence,
            "bbox": result.get("bbox"),
        }

    def target_is_recent(self):
        if self.last_target_detection is None:
            return False
        return (rospy.Time.now() - self.last_target_detection["time"]).to_sec() <= self.detection_timeout

    def classifier_is_ready(self):
        if not self.require_classifier_ready:
            return True
        if self.last_classifier_result_time is None:
            return False
        return (rospy.Time.now() - self.last_classifier_result_time).to_sec() <= self.classifier_timeout

    def control_loop(self, _event):
        if self.stopped:
            self.stop_robot()
            return

        if not self.classifier_is_ready():
            self.stop_robot()
            rospy.logwarn_throttle(
                2.0,
                "Waiting for camera/classifier results on %s before moving.",
                self.result_topic,
            )
            return

        if math.isfinite(self.front_distance) and self.front_distance <= self.stop_distance and self.target_is_recent():
            self.stopped = True
            self.stop_robot()
            rospy.loginfo("Target %s reached. Stopping at %.2f m.", self.target_symbol, self.front_distance)
            return

        cmd = Twist()
        if self.target_is_recent():
            center_error = self.last_target_detection["center_x"] - 0.5
            cmd.angular.z = self.clamp(-self.align_kp * center_error, -self.max_angular_speed, self.max_angular_speed)
            if abs(center_error) <= self.center_deadband:
                cmd.linear.x = self.approach_speed
            else:
                cmd.linear.x = self.slow_approach_speed
            rospy.loginfo_throttle(
                1.0,
                "Approaching %s: confidence=%.2f center_error=%.2f front=%.2f",
                self.target_symbol,
                self.last_target_detection["confidence"],
                center_error,
                self.front_distance,
            )
        else:
            cmd.angular.z = self.search_angular_speed
            rospy.loginfo_throttle(2.0, "Searching for target symbol: %s", self.target_symbol)

        if self.enable_motion:
            self.cmd_pub.publish(cmd)

    def stop_robot(self):
        if self.enable_motion:
            self.cmd_pub.publish(Twist())


if __name__ == "__main__":
    try:
        TargetSymbolController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
