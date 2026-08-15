#!/usr/bin/env python3
"""
Simple reactive obstacle avoidance for TortoiseBot (ROS2 Humble).

Subscribes:  /scan  (sensor_msgs/LaserScan)
Publishes:   /cmd_vel (geometry_msgs/Twist)

Behavior:
  - Drive forward while the front sector is clear.
  - If an obstacle is closer than `stop_distance` in front, stop forward
    motion and rotate toward the side (left/right) with more free space.
  - Keep rotating until the front is clear again, then resume forward.

Works with both the Gazebo lidar (angles -pi..pi) and the real YDLidar
(angles 0..2pi) because sectors are computed from each beam's actual angle,
not from array indices.

Run (after the simulation or real robot bringup is up):
    ros2 run tortoisebot_control obstacle_avoidance
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class ObstacleAvoidance(Node):

    def __init__(self):
        super().__init__('obstacle_avoidance')

        # ---------------- Parameters (overridable from CLI/launch) ----------------
        self.declare_parameter('stop_distance', 0.4)    # [m] obstacle closer than this -> turn
        self.declare_parameter('clear_distance', 0.5)   # [m] front must be clearer than this to resume (hysteresis)
        self.declare_parameter('forward_speed', 0.12)   # [m/s] cruise speed (robot max ~0.15)
        self.declare_parameter('turn_speed', 0.6)       # [rad/s] rotation speed while avoiding
        self.declare_parameter('front_angle_deg', 30.0) # front sector = +/- this angle
        self.declare_parameter('side_angle_deg', 90.0)  # side sectors extend from front sector to this angle

        self.stop_distance = self.get_parameter('stop_distance').value
        self.clear_distance = self.get_parameter('clear_distance').value
        self.forward_speed = self.get_parameter('forward_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value
        self.front_angle = math.radians(self.get_parameter('front_angle_deg').value)
        self.side_angle = math.radians(self.get_parameter('side_angle_deg').value)

        # ---------------- State ----------------
        self.turning = False       # currently in avoidance turn
        self.turn_direction = 1.0  # +1 = left (CCW), -1 = right (CW)

        # ---------------- ROS interfaces ----------------
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)

        self.get_logger().info(
            f'Obstacle avoidance started: stop at {self.stop_distance} m, '
            f'forward {self.forward_speed} m/s, turn {self.turn_speed} rad/s')

    # ------------------------------------------------------------------
    @staticmethod
    def normalize_angle(angle):
        """Wrap angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def sector_min(self, msg, valid):
        """Return (front_min, left_min, right_min) distances from the scan.

        front: |angle| <= front_angle
        left:  front_angle < angle <= side_angle      (CCW, robot's left)
        right: -side_angle <= angle < -front_angle    (CW, robot's right)
        """
        front = left = right = float('inf')
        for i, r in enumerate(msg.ranges):
            if not valid(r):
                continue
            angle = self.normalize_angle(msg.angle_min + i * msg.angle_increment)
            if abs(angle) <= self.front_angle:
                front = min(front, r)
            elif self.front_angle < angle <= self.side_angle:
                left = min(left, r)
            elif -self.side_angle <= angle < -self.front_angle:
                right = min(right, r)
        return front, left, right

    # ------------------------------------------------------------------
    def scan_callback(self, msg):
        range_min = max(msg.range_min, 0.05)  # ignore self-hits / dead zone
        range_max = msg.range_max if msg.range_max > 0.0 else 20.0

        def valid(r):
            return (not math.isnan(r)) and (not math.isinf(r)) and range_min < r < range_max

        front, left, right = self.sector_min(msg, valid)

        cmd = Twist()

        if self.turning:
            # Keep turning until front is clear (with hysteresis so we
            # don't oscillate right at the threshold).
            if front > self.clear_distance:
                self.turning = False
                cmd.linear.x = self.forward_speed
                self.get_logger().info(f'Path clear (front {front:.2f} m) -> forward')
            else:
                cmd.angular.z = self.turn_direction * self.turn_speed
        else:
            if front < self.stop_distance:
                # Obstacle ahead: pick the side with more room and turn.
                self.turning = True
                self.turn_direction = 1.0 if left >= right else -1.0
                side = 'left' if self.turn_direction > 0 else 'right'
                self.get_logger().info(
                    f'Obstacle at {front:.2f} m -> turning {side} '
                    f'(left {left:.2f} m, right {right:.2f} m)')
                cmd.angular.z = self.turn_direction * self.turn_speed
            else:
                cmd.linear.x = self.forward_speed
                # Slow down as we approach an obstacle for a smoother stop.
                if front < 2.0 * self.stop_distance:
                    cmd.linear.x = max(0.05, self.forward_speed * (front - self.stop_distance)
                                       / self.stop_distance)

        self.cmd_pub.publish(cmd)

    # ------------------------------------------------------------------
    def stop_robot(self):
        """Publish a zero Twist so the robot doesn't keep driving on shutdown."""
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
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
