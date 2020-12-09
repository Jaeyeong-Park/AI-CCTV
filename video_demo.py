#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import time

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
# video_path      = "./docs/images/road.mp4"
#video_path      = "C:/Users/wodud/Videos/Logitech/LogiCapture/test_5.mp4"
video_path      = 0
num_classes     = 80
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)



with tf.Session(graph=graph) as sess:
    prevTime=0
    vid = cv2.VideoCapture(video_path)

    while True:
        return_value, frame = vid.read()
        curTime=time.time()
        sec=curTime-prevTime
        prevTime=curTime
        fps=1/(sec)
        str="FPS:%0.1f"%fps
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            if utils.select == 1:
                video_path = "C:/Users/user/Videos/Logitech/LogiCapture/test_5.mp4"
                utils.select = 0
                ddd = {}
                ddd_num = {}
                utils.invade = 0

            else:
                video_path = "C:/Users/user/Videos/Logitech/LogiCapture/test_6.mp4"
                utils.select = 1
                ddd = {}
                ddd_num = {}
                utils.k = 0
            # raise ValueError("No image!")
            vid = cv2.VideoCapture(video_path)
            return_value, frame = vid.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]


        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame, bboxes)


        result = np.asarray(image)

        cv2.putText(result, str, (0, 460), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if utils.select == 0:
            cv2.line(result, (48, 245), (173, 289), (0, 255, 0))
            cv2.line(result, (173, 289), (201, 268), (0, 255, 0))
            cv2.line(result, (201, 268), (332, 322), (0, 255, 0))
            cv2.line(result, (332, 322), (336, 398), (0, 255, 0))
            cv2.line(result, (336, 398), (43, 398), (0, 255, 0))
            cv2.line(result, (43, 398), (48, 245), (0, 255, 0))
            cv2.line(result, (211, 255), (290, 278), (0, 0, 255))
            cv2.line(result, (290, 278), (290, 303), (0, 0, 255))
            cv2.line(result, (290, 303), (203, 268), (0, 0, 255))
            cv2.line(result, (203, 268), (211, 255), (0, 0, 255))
            cv2.putText(result, '<Warning Area>', (480, 446), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
            cv2.putText(result, '<Prohibited Line>', (470, 469), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0,255))

        cv2.imshow("result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break





