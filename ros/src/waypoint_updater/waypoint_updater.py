#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''


class WaypointUpdater(object):

    LOOKAHEAD_WPS = 50
    MAX_BREAKING = 1.0
    MPS_2_MPH = 2.23694
    SAFETY_COEFF = 0.90

    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stop_wp_index = -1

        self.loop()

    def loop(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                self.publish_waypoints(self.nearest_waypoint_index())
            rate.sleep()

    def nearest_waypoint_index(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        nearest_index = self.waypoint_tree.query([x, y], 1)[1]
        nearest_coord = self.waypoints_2d[nearest_index]
        prev_coord = self.waypoints_2d[nearest_index - 1]
        dot = np.dot(np.array(nearest_coord) - np.array(prev_coord), np.array([x, y]) - np.array(nearest_coord))

        if dot > 0:
            nearest_index = (nearest_index + 1) % len(self.waypoints_2d)

        return nearest_index

    def publish_waypoints(self, nearest_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header

        if self.stop_wp_index == -1 or (self.stop_wp_index >= nearest_idx + WaypointUpdater.LOOKAHEAD_WPS):
            lane.waypoints = self.base_waypoints.waypoints[nearest_idx:nearest_idx + WaypointUpdater.LOOKAHEAD_WPS]
        else:
            lane.waypoints = self.decelerate_waypoints(lane.waypoints, nearest_idx)

        self.final_waypoints_pub.publish(lane)

    def decelerate_waypoints(self, waypoints, nearest_index):
        points = []
        for i, wp in enumerate(waypoints):
            new_point = Waypoint()
            new_point.pose = wp.pose

            stop_index = max(self.stop_wp_index - nearest_index - 2, 0)
            dist = WaypointUpdater.distance(waypoints, i, stop_index)
            vel = math.sqrt(2 * WaypointUpdater.MAX_BREAKING * WaypointUpdater.SAFETY_COEFF * dist)
            if vel < 1.0:
                vel = 0.0

            new_point.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            points.append(new_point)

        return points

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,
                                  waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stop_wp_index = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    @staticmethod
    def distance(waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
