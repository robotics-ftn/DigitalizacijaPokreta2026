import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from pathlib import Path


def record(img_name, vid_name, rate=15, size=(1920, 1080)):
    context = rs.context()
    pipelines = list()
    configs = list()
    serials = list()
    for device in context.query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color, size[0], size[1], rs.format.bgr8, rate)
        config.enable_device(serial)
        configs.append(config)
        pipelines.append(pipeline)
        serials.append(serial)

    for config, pipeline in zip(configs, pipelines):
        pipeline.start(config)
    try:
        k = 0
        img_num = 0
        vid_num = 0
        save_image = False
        save_video = False
        cv2.namedWindow('RealSense', cv2.WINDOW_KEEPRATIO)
        while k != 27:  # not escape
            images = list()
            k = cv2.waitKey(1)

            if (k == 32):
                save_image = True
            if (k == 86 or k == 118):
                if not save_video:
                    video_writers = list()
                    for serial in serials:
                        name = vid_name.format(serial=serial, num=vid_num)
                        Path(name).parent.mkdir(parents=True, exist_ok=True)

                        writer = cv2.VideoWriter(
                            name, cv2.VideoWriter_fourcc(*'mp4v'), rate, size)
                        video_writers.append(writer)
                    save_video = True
                else:
                    for writer in video_writers:
                        writer.release()
                    save_video = False
                    vid_num = vid_num + 1

            for pipeline in pipelines:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                images.append(np.asanyarray(color_frame.get_data()))

            if save_image:
                for img, serial in zip(images, serials):
                    name = img_name.format(serial=serial, num=img_num)
                    Path(name).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(name, img)
                save_image = False
                img_num = img_num + 1

            if save_video:
                for img, writer in zip(images, video_writers):
                    writer.write(img)
            # Show images
            color_image = np.hstack(images)
            if save_video:
                cv2.circle(color_image, (100, 100), 50, (0, 0, 255), -1)

            # Put text
            color = (0, 0, 255)
            text = "SPACE: save img"
            cv2.putText(color_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            text = "V: start/stop video"
            cv2.putText(color_image, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            text = "ESC: exit"
            cv2.putText(color_image, text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            cv2.imshow('RealSense', color_image)
    finally:
        pass

    for pipeline in pipelines:
        pipeline.stop()


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + "/data/Lazar/Squat/"

    vid_name = "{serial}/vid_{num:05d}.mp4"
    img_name = "{serial}/img_{num:05d}.png"

    vid_name = os.path.join(path + "/video", vid_name)
    img_name = os.path.join(path + "/images", img_name)
    record(img_name, vid_name)
