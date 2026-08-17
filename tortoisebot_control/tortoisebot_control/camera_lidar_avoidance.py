#!/usr/bin/env python3
"""
Camera + LiDAR fused obstacle avoidance for TortoiseBot (ROS2 Humble).

Subscribes:  /scan               (sensor_msgs/LaserScan)  - primary distance sensing
             image topic         (sensor_msgs/Image)      - catches low obstacles the
                                                            lidar plane can miss
Publishes:   /cmd_vel            (geometry_msgs/Twist)

Fusion logic (runs in a 10 Hz control loop):
  - LiDAR gives min distance in front / left / right sectors (by beam angle).
  - Camera: the bottom half of the image (the floor ahead) is split into
    left / center / right thirds. Canny edge density in each third is a
    cheap "something is there" detector - an empty floor is smooth (low
    edge density), an object close ahead produces many edges.
  - The robot treats the path as BLOCKED if the lidar front distance is
    below stop_distance OR the camera center region is busy while the
    lidar already shows something within caution range (2 x stop_distance).
  - Turn direction: the side with more lidar free space; camera edge
    density is the tie-breaker.
  - If the camera never publishes (not started / wrong topic), the node
    keeps working as lidar-only and logs a warning once.
  - If the LIDAR goes stale (> 1 s without a scan), the robot stops. The
    camera alone is never trusted to drive.

Run:
    ros2 run tortoisebot_control camera_lidar_avoidance
or
    ros2 launch tortoisebot_control camera_lidar_avoidance.launch.py

Check your actual image topic with:  ros2 topic list | grep -i image
and override if needed:
    ros2 launch tortoisebot_control camera_lidar_avoidance.launch.py image_topic:=/camera/image_raw
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist

import numpy as np

try:
    import cv2
    from cv_bridge import CvBridge
    CV_AVAILABLE = True
except ImportError:  # camera support becomes a no-op, lidar still works
    CV_AVAILABLE = False


class CameraLidarAvoidance(Node):

    def __init__(self):
        super().__init__('camera_lidar_avoidance')

        # ---------------- Parameters ----------------
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('stop_distance', 0.4)     # [m] lidar front closer -> turn
        self.declare_parameter('clear_distance', 0.5)    # [m] front must exceed this to resume
        self.declare_parameter('forward_speed', 0.12)    # [m/s]
        self.declare_parameter('turn_speed', 0.6)        # [rad/s]
        self.declare_parameter('front_angle_deg', 30.0)
        self.declare_parameter('side_angle_deg', 90.0)
        self.declare_parameter('edge_density_threshold', 0.12)  # 0..1, camera "busy" level
        self.declare_parameter('scan_timeout', 1.0)      # [s] stop if lidar older than this

        self.image_topic = self.get_parameter('image_topic').value
        self.stop_distance = self.get_parameter('stop_distance').value
        self.clear_distance = self.get_parameter('clear_distance').value
        self.forward_speed = self.get_parameter('forward_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value
        self.front_angle = math.radians(self.get_parameter('front_angle_deg').value)
        self.side_angle = math.radians(self.get_parameter('side_angle_deg').value)
        self.edge_threshold = self.get_parameter('edge_density_threshold').value
        self.scan_timeout = self.get_parameter('scan_timeout').value

        # ---------------- Sensor state ----------------
        self.front = self.left = self.right = float('inf')
        self.last_scan_time = None
        self.cam_left = self.cam_center = self.cam_right = 0.0
        self.camera_seen = False
        self._warned_no_camera = False

        # ---------------- Control state ----------------
        self.turning = False
        self.turn_direction = 1.0  # +1 left (CCW), -1 right (CW)

        # ---------------- ROS interfaces ----------------
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)

        if CV_AVAILABLE:
            self.bridge = CvBridge()
            self.image_sub = self.create_subscription(
                Image, self.image_topic, self.image_callback, qos_profile_sensor_data)
        else:
            self.get_logger().warn(
                'cv_bridge / OpenCV not available - running LIDAR-ONLY. '
                'Install with: sudo apt install ros-humble-cv-bridge python3-opencv')

        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.get_logger().info(
            f'Camera+LiDAR avoidance started (image topic: {self.image_topic}, '
            f'stop {self.stop_distance} m, edge threshold {self.edge_threshold})')

    # ------------------------------------------------------------------
    #  LiDAR
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_angle(angle):
        """Wrap angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def scan_callback(self, msg):
        range_min = max(msg.range_min, 0.05)
        range_max = msg.range_max if msg.range_max > 0.0 else 20.0

        front = left = right = float('inf')
        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r) or not (range_min < r < range_max):
                continue
            angle = self.normalize_angle(msg.angle_min + i * msg.angle_increment)
            if abs(angle) <= self.front_angle:
                front = min(front, r)
            elif self.front_angle < angle <= self.side_angle:
                left = min(left, r)
            elif -self.side_angle <= angle < -self.front_angle:
                right = min(right, r)

        self.front, self.left, self.right = front, left, right
        self.last_scan_time = self.get_clock().now()

    # ------------------------------------------------------------------
    #  Camera
    # ------------------------------------------------------------------
    def image_callback(self, msg):
        self.camera_seen = True
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Image conversion failed: {exc}',
                                   throttle_duration_sec=5.0)
            return

        h, w = frame.shape[:2]
        # Bottom half of the image = floor directly ahead of the robot.
        roi = frame[h // 2:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        third = w // 3
        left_roi = edges[:, :third]
        center_roi = edges[:, third:2 * third]
        right_roi = edges[:, 2 * third:]

        # Edge density: fraction of pixels that are edges (0..1).
        self.cam_left = float(np.count_nonzero(left_roi)) / left_roi.size
        self.cam_center = float(np.count_nonzero(center_roi)) / center_roi.size
        self.cam_right = float(np.count_nonzero(right_roi)) / right_roi.size

    # ------------------------------------------------------------------
    #  Fusion + control (10 Hz)
    # ------------------------------------------------------------------
    def control_loop(self):
        cmd = Twist()

        # Safety: no (or stale) lidar -> stand still. Never drive on camera alone.
        now = self.get_clock().now()
        if (self.last_scan_time is None or
                (now - self.last_scan_time).nanoseconds * 1e-9 > self.scan_timeout):
            self.cmd_pub.publish(cmd)
            self.get_logger().warn('No recent /scan data - robot stopped',
                                   throttle_duration_sec=5.0)
            return

        if CV_AVAILABLE and not self.camera_seen and not self._warned_no_camera:
            self._warned_no_camera = True
            self.get_logger().warn(
                f'No images on {self.image_topic} yet - running lidar-only. '
                'Check the topic with: ros2 topic list | grep -i image')

        camera_blocked = (self.camera_seen and
                          self.cam_center > self.edge_threshold and
                          self.front < 2.0 * self.stop_distance)
        lidar_blocked = self.front < self.stop_distance
        blocked = lidar_blocked or camera_blocked

        if self.turning:
            camera_clear = (not self.camera_seen) or (self.cam_center <= self.edge_threshold)
            if self.front > self.clear_distance and camera_clear:
                self.turning = False
                cmd.linear.x = self.forward_speed
                self.get_logger().info(
                    f'Path clear (front {self.front:.2f} m, '
                    f'cam {self.cam_center:.2f}) -> forward')
            else:
                cmd.angular.z = self.turn_direction * self.turn_speed
        elif blocked:
            self.turning = True
            self.turn_direction = self.pick_turn_direction()
            side = 'left' if self.turn_direction > 0 else 'right'
            source = 'lidar' if lidar_blocked else 'camera'
            self.get_logger().info(
                f'Blocked by {source} (front {self.front:.2f} m, '
                f'cam L/C/R {self.cam_left:.2f}/{self.cam_center:.2f}/{self.cam_right:.2f}) '
                f'-> turning {side}')
            cmd.angular.z = self.turn_direction * self.turn_speed
        else:
            cmd.linear.x = self.forward_speed
            # Ease off as we approach whatever the lidar sees ahead.
            if self.front < 2.0 * self.stop_distance:
                cmd.linear.x = max(0.05, self.forward_speed *
                                   (self.front - self.stop_distance) / self.stop_distance)

        self.cmd_pub.publish(cmd)

    def pick_turn_direction(self):
        """+1 = turn left, -1 = turn right. Lidar decides; camera breaks ties."""
        if math.isfinite(self.left) or math.isfinite(self.right):
            l = self.left if math.isfinite(self.left) else 1e6
            r = self.right if math.isfinite(self.right) else 1e6
            if abs(l - r) > 0.05:  # clear lidar winner
                return 1.0 if l >= r else -1.0
        if self.camera_seen and abs(self.cam_left - self.cam_right) > 0.02:
            # Fewer edges = emptier view -> more likely free space.
            return 1.0 if self.cam_left <= self.cam_right else -1.0
        return 1.0  # default: left

    # ------------------------------------------------------------------
    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = CameraLidarAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
