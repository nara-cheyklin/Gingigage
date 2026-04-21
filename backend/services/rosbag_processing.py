import numpy as np
import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


def ros_image_to_cv2(msg: Image):
    if msg.encoding == "rgb8":
        img = np.frombuffer(msg.data, dtype=np.uint8)
        img = img.reshape((msg.height, msg.width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif msg.encoding == "bgr8":
        img = np.frombuffer(msg.data, dtype=np.uint8)
        return img.reshape((msg.height, msg.width, 3))

    elif msg.encoding == "16UC1":
        img = np.frombuffer(msg.data, dtype=np.uint16)
        return img.reshape((msg.height, msg.width))

    elif msg.encoding == "32FC1":
        img = np.frombuffer(msg.data, dtype=np.float32)
        return img.reshape((msg.height, msg.width))

    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def extract_rgb_and_depth_from_rosbag(
    bag_path: str,
    rgb_topic: str = "/camera/color/image_raw",
    depth_topic: str = "/camera/aligned_depth_to_color/image_raw"
):
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3"
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    rgb_frame = None
    depth_frame = None

    while reader.has_next():
        topic, data, _ = reader.read_next()

        if topic == rgb_topic and rgb_frame is None:
            msg = deserialize_message(data, Image)
            rgb_frame = ros_image_to_cv2(msg)

        elif topic == depth_topic and depth_frame is None:
            msg = deserialize_message(data, Image)
            depth_frame = ros_image_to_cv2(msg)

        if rgb_frame is not None and depth_frame is not None:
            break

    if rgb_frame is None:
        raise RuntimeError("No RGB frame found in ROSBAG")

    if depth_frame is None:
        raise RuntimeError("No aligned depth frame found in ROSBAG")

    return rgb_frame, depth_frame


def cv2_to_bytes(image):
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Failed to encode RGB image")
    return buffer.tobytes()