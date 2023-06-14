#!/usr/bin/env python3

import rospy
import numpy as np
from filterpy.kalman import KalmanFilter
from geometry_msgs.msg import Point, TwistStamped, PoseStamped


class FusionNode:
    def __init__(self, node_name="kalman_filter", px4_position_topic="/mavros/local_position/pose",
                 vel_topic="/mavros/setpoint_velocity/cmd_vel", vision_topic="/uav/marker/corrected_position",
                 fused_pose_topic="/uav/fused_pose"):  # fused_vision_topic='/uav/marker/fused_position'
        rospy.init_node(node_name, anonymous=True)

        # Subscribers
        self.px4_position_sub = rospy.Subscriber(px4_position_topic, PoseStamped, self.px4_position_cb)
        self.px4_velocity_sub = rospy.Subscriber(vel_topic, TwistStamped, self.px4_velocity_cb)
        self.aruco_info_sub = rospy.Subscriber(vision_topic, Point, self.vision_position_cb)

        # Publishers
        # self.pub = rospy.Publisher(fused_pose_topic, PoseStamped, self.fused_msg, queue_size=10)
        self.fused_pose_pub = rospy.Publisher(fused_pose_topic, PoseStamped, queue_size=10)

        self.px4_position = None
        self.vision_position = None
        self.px4_velocity = None
        # self.imu_velocity = None
        # self.current_estimate = None
        # self.fused_msg = PoseStamped()
        rate = 20
        self.rate = rospy.Rate(rate)

        # Initialize the main Kalman filter
        self.kf = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
        self.kf.x = np.array([[0], [0], [0]])
        self.kf.H = np.eye(3)
        self.kf.P *= 1

        # self.kf.Q = np.array([[0.0001, 0, 0],
        #                       [0, 0.0001, 0],
        #                       [0, 0, 0.0001]])

        self.kf.Q = np.array([[0.01, 0, 0],
                              [0, 0.01, 0],
                              [0, 0, 0.01]])

        self.kf.B = np.array([[1 / rate, 0, 0],
                              [0, 1 / rate, 0],
                              [0, 0, 1 / rate]])

        # Init velocity inputs
        self.kf.u = np.array([[0], [0], [0]])

    def px4_position_cb(self, msg):
        self.px4_position = np.array([[msg.pose.position.x],
                                      [msg.pose.position.y],
                                      [msg.pose.position.z]])

        # self.kf.R = np.array([[0.01, 0, 0],
        #                       [0, 0.01, 0],
        #                       [0, 0, 0.01]])

        self.kf.R = np.array([[0.25, 0, 0],
                              [0, 0.25, 0],
                              [0, 0, 0.25]])

        self.kf.update(self.px4_position, self.kf.R, self.kf.H)

        # rospy.loginfo('px4 position is %s', self.px4_position)

    def px4_velocity_cb(self, msg):
        self.kf.u = np.array([[msg.twist.linear.x],
                              [msg.twist.linear.y],
                              [msg.twist.linear.z]])

        # rospy.loginfo('px4 velocity is %s', self.kf.u)

    def vision_position_cb(self, msg):
        self.vision_position = np.array([[msg.x],
                                         [msg.y],
                                         [msg.z]])

        self.kf.R = np.array([[0.01, 0, 0],
                              [0, 0.01, 0],
                              [0, 0, 0.01]])

        self.kf.update(self.vision_position, self.kf.R, self.kf.H)

        # rospy.loginfo('vision position is %s', self.vision_position)

    def fuse_poses(self):
        self.kf.predict(self.kf.u, self.kf.B, self.kf.F, self.kf.Q)

        # Publish the fused pose
        fused_pose = PoseStamped()
        fused_pose.header.stamp = rospy.Time.now()
        fused_pose.header.frame_id = "fused_frame"
        fused_pose.pose.position.x = self.kf.x[0]
        fused_pose.pose.position.y = self.kf.x[1]
        fused_pose.pose.position.z = self.kf.x[2]

        # rospy.loginfo('fused message is %s', fused_pose.pose.position)
        self.fused_pose_pub.publish(fused_pose)

    def run(self):
        while not rospy.is_shutdown():
            self.fuse_poses()
            self.rate.sleep()


def main():
    try:
        data_fusion = FusionNode()
        data_fusion.run()
        rospy.spin()
    except rospy.ROSInterruptException as exception:
        pass


if __name__ == '__main__':
    main()
