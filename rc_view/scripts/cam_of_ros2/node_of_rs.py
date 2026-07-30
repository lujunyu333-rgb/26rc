#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rc_view.scripts.cam_of_ros2.cam_ros import RealSensePublisher

def main(args=None):
    rclpy.init(args=args)
    node = RealSensePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()