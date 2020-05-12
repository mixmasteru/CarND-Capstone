#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import tensorflow
import yaml


class TLDetector(object):

    STATE_THRESHOLD = 2

    def __init__(self):
        rospy.init_node('tl_detector')
        self.frame = 0
        self.frame_d = 5
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints_2d = None
        self.waypoint_tree = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1, buff_size=2 * 52428800)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.next_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        is_site = self.config['is_site']
        graph_file = 'my_model_final_styx.pb'
        if is_site:
            graph_file = 'my_model_final_site.pb'

        self.graph = tensorflow.Graph()

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.graph, graph_file)
        self.listener = tf.TransformListener()
        self.sess = tensorflow.Session(graph=self.graph)

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''

        if light_wp == -1 and state == TrafficLight.UNKNOWN:
            return

        try:
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= TLDetector.STATE_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.next_red_light_pub.publish(Int32(light_wp))
            else:
                self.next_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

        except Exception as e:
            print "error in tl_detector " + e.message
            return

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        min_wp = -1
        if pose is not None and self.waypoint_tree is not None:
            min_wp = self.waypoint_tree.query([pose.position.x, pose.position.y], 1)[1]

        return min_wp

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        try:
            classification = self.light_classifier.get_classification(self.sess, cv_image)
        except Exception as e:
            print "error in tl_classifier " + e.message
            return False

        return classification

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        self.frame += 1
        if self.frame % self.frame_d is not 0:
            return -1, TrafficLight.UNKNOWN
        self.frame = 0

        light = None
        closest_ls_wp = None
        dist_to_light = 10000
        stop_line_positions = self.config['stop_line_positions']
        if self.pose:
            car_position = self.get_closest_waypoint(self.pose.pose)
        else:
            return -1, TrafficLight.UNKNOWN

        for stop_line_position in stop_line_positions:
            sl_pose = Pose()
            sl_pose.position.x = stop_line_position[0]
            sl_pose.position.y = stop_line_position[1]

            ls_wp = self.get_closest_waypoint(sl_pose)

            if ls_wp >= car_position:
                if closest_ls_wp is None or ls_wp < closest_ls_wp:
                    closest_ls_wp = ls_wp
                    light = sl_pose
                if car_position is not None and closest_ls_wp is not None:
                    dist_to_light = abs(car_position - closest_ls_wp)

        if light and dist_to_light < 80:
            state = self.get_light_state(light)
            return closest_ls_wp, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
