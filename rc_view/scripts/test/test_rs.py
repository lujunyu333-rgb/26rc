from allcamera import rs_cam
import cv2

if __name__ == "__main__":
    realsense_cam=rs_cam()
    try :
        while True:
            color_image, depth_image = realsense_cam.read_one_frame()
            if color_image is None or depth_image is None:
                continue
            cv2.imshow("Realsense",color_image)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 键退出
                break
    finally:
        realsense_cam.close_rscam()
