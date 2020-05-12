from styx_msgs.msg import TrafficLight
import cv2
import tensorflow as tf
import numpy as np


class TLClassifier(object):
    def __init__(self, graph, graph_file):
        self.graph = graph

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    @staticmethod
    def create_feature(img_rgb):
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        brightness_channel = hsv[:, :, 2]
        rows = brightness_channel.shape[0]
        cols = brightness_channel.shape[1]
        mid = int(cols / 2)

        red = brightness_channel[:int(rows / 3), (mid - 10):(mid + 10)]
        yellow = brightness_channel[int(rows / 3):int(2 * rows / 3), (mid - 10):(mid + 10)]
        green = brightness_channel[int(2 * rows / 3):, (mid - 10):(mid + 10)]

        return [np.mean(green), np.mean(red), np.mean(yellow)]

    def predict_traffic_class(self, traffic_img):
        standard_im = cv2.resize(np.copy(traffic_img), (32, 32))
        rgb_img = cv2.cvtColor(standard_im, cv2.COLOR_BGR2RGB)
        max_index = np.argmax(self.create_feature(rgb_img)) + 1.

        return max_index

    def detect_traffic_light(self, sess, image_tensor, detect_boxes, detect_scores, detect_classes, num_detections,
                             img_bgr, min_score_thresh):

        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channels = image.shape
        image_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes, num) = sess.run([detect_boxes, detect_scores, detect_classes, num_detections],
                                                 feed_dict={image_tensor: image_expanded})
        classes[classes == 2.] = 4.

        for i in range(boxes.shape[1]):
            if scores[0, i] > min_score_thresh:
                y_min = int(boxes[0, i, 0] * img_height)
                x_min = int(boxes[0, i, 1] * img_width)
                y_max = int(boxes[0, i, 2] * img_height)
                x_max = int(boxes[0, i, 3] * img_width)

                if ((y_max - y_min) * (x_max - x_min)) >= 1024:
                    traffic_img = image[y_min:y_max, x_min:x_max]
                    classes[0, i] = self.predict_traffic_class(traffic_img)
            else:
                break

        return boxes, scores, classes, num, image

    def get_classification(self, sess, image):
        with self.graph.as_default():

            image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            detect_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            detect_scores = self.graph.get_tensor_by_name('detection_scores:0')
            detect_classes = self.graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.graph.get_tensor_by_name('num_detections:0')

            min_score_thresh = .6
            (boxes, scores, classes, num, image_np) = self.detect_traffic_light(sess, image_tensor, detect_boxes,
                                                                                detect_scores, detect_classes,
                                                                                num_detections, image, min_score_thresh)

            classification = classes[0][0]
            score = scores[0][0]

            if classification == 1.0:
                print('GREEN - ', score)
                return TrafficLight.GREEN
            elif classification == 2.0:
                print('RED - ', score)
                return TrafficLight.RED
            elif classification == 3.0:
                print('YELLOW - ', score)
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
