#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : utils.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:14:19
#   Description :
#
#================================================================

import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.config import cfg
##########
global ddd                 #객체의 (x, y, w, h) 값을 갖는 dictionary 변수
global ddd_num             #객체의 파라미터 값을 갖는 dictionary 변수
global img_gal             #이전 프레임 이미지
invade = 0                 #침입시 'INVADE' 문자열 반짝이게 하기위한 변수
k = 0                      #배회시 검출되는 객체의 키값을 나타내기위한 변수
select = 0                 #두가지 동영상을 구분하기위한 변수
ddd = {}
ddd_num = {}
img_gal = 0
count = 0

##########

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preporcess(image, target_size, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes



def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    ##########
    global img_gal           #각 전역변수를 불러오기
    global k
    global select
    global invade
    global count

    roi_img = cv2.imread("C:/Users/user/Videos/Logitech/LogiCapture/img_sample.png", cv2.IMREAD_GRAYSCALE)  #ROI설정 이미지 불러오기
    img_now = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                                                        #현재 프레임 이미지를 탬플릿 매칭을 위하여 BGR 이미지로 불러옴
    ddd_num[0] = [0, 0]                                                                                     #탬플릿 매칭이 처음 실행될 때 비교하기위한 기준 값
    ##########
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):

        class_ind = int(bbox[5])

        if show_label and classes[class_ind] == 'person':
            coor = np.array(bbox[:4], dtype=np.int32)
            fontScale = 0.5
            score = bbox[4]

            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            ##########
            p_x = coor[0]           #검출된 객체의 (x, y, w, h) 값
            p_y = coor[1]
            p_h = coor[3] - p_y
            p_w = coor[2] - p_x

            bbox_mess = '%s: %.2f' % (classes[class_ind], score) #있던거

            if type(img_gal) == int:               #처음 실행시 이전 이미지가 없을 것을 방지하기 위한 조건
                if [p_x, p_y, p_w, p_h] not in ddd.values():  # 일치하는 이미지 없을시 새로 저장
                    for j in range(1, len(ddd) + 2):
                        if j not in ddd.keys():
                            ddd[j] = [p_x, p_y, p_w, p_h]
                            ddd_num[j] = [0, 0,0,0]
                            break
            else:
                if len(ddd):
                    sum_tem = [0, 1]                      #  템플릿 최소값 갖기 위한 변수
                    for a, b in ddd.items():
                        th1 = ddd[a][3]  # h1
                        tw1 = ddd[a][2]  # w1
                        th2 = p_h  # h2
                        tw2 = p_w  # w2
                        if (ddd[a][2] < p_w) and (ddd[a][3] > p_h):  # w1 < w2  and h1 > h2
                            if abs(p_w - ddd[a][2]) > abs(p_h > ddd[a][3]):  # >
                                th1 = p_h  # h 변화
                            elif abs(p_w - ddd[a][2]) <= abs(p_h > ddd[a][3]):
                                tw2 = ddd[a][2]
                        elif (p_w < ddd[a][2]) and (p_h > ddd[a][3]):  # w1 > w2  and h1 < h2
                            if abs(p_w - ddd[a][2]) > abs(p_h > ddd[a][3]):  # >
                                th2 = ddd[a][3]  # h 변화
                            elif abs(p_w - ddd[a][2]) <= abs(p_h > ddd[a][3]):
                                tw1 = p_w
                        img_tem1 = img_gal[ddd[a][1]:ddd[a][1] + th1, ddd[a][0]:ddd[a][0] + tw1]
                        img_tem2 = img_now[p_y:p_y + th2, p_x:p_x + tw2]                          #두 이미지를 서로 같게 만듬

                        if (abs(ddd[a][1] - p_y) < 30) or (abs(ddd[a][0] - p_x) < 30):                 #x, y좌표값이 너무 차이나면 실행하지 않음
                            res = cv2.matchTemplate(img_tem1, img_tem2, cv2.TM_SQDIFF_NORMED)
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                            if (min_val < 0.2) and (ddd_num[a][0] >= ddd_num[sum_tem[0]][0]):  #잘된게 두가지가 있을경우 매칭 수가 많은것을 채택
                                if min_val < sum_tem[1]:       #매칭값이 0.2보다 낮은걸 가져오되 중간에 튀었으면 값이 큰것을 가져옴
                                    sum_tem[1] = min_val
                                    sum_tem[0] = a
                    if sum_tem[1] < 0.2:
                        ddd[sum_tem[0]] = [p_x, p_y, p_w, p_h]  # 템플릿 잘된거에 이미지 넣기
                        ddd_num[sum_tem[0]][0] += 1  # 매칭된 수 +1
                        ddd_num[sum_tem[0]][1] = 0   # 매칭 안된수 초기화
                        k = sum_tem[0]
                        if ddd_num[sum_tem[0]][2] > 50:   # 매칭된수 50회 넘어갈시 노랑색으로 표시
                            bbox_mess = '[ %d ] Stranger : %.2f' % (k, score)
                            bbox_color = (255, 255, 0)

                if [p_x, p_y, p_w, p_h] not in ddd.values():  # 탬플릿 매칭이 안될 시 새로 저장
                    for j in range(1, len(ddd)+2):
                        if j not in ddd.keys():
                            ddd[j] = [p_x, p_y, p_w, p_h]
                            ddd_num[j] = [0, 0,0,0]
                            break

            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0] #있던거

            if select == 0:             #동영상 별 글씨 및 바운딩 박스
                cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled
                cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
                cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0),
                        bbox_thick // 2, lineType=cv2.LINE_AA)
            else:

                cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), (255, 0, 0), -1)  # filled
                cv2.rectangle(image, c1, c2, (255, 0, 0), bbox_thick+1)
                cv2.putText(image, 'stranger', (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0),
                        bbox_thick // 2, lineType=cv2.LINE_AA)

                img_trim=image[p_y:p_y+p_h,p_x:p_x+p_w]
                img_trim = cv2.cvtColor(img_trim, cv2.COLOR_RGB2BGR)
                cv2.imwrite('./Invade image/stranger'+str(count)+'.jpg',img_trim)
                if count == 100:
                      count = 0
                count+=1



    img_gal = img_now.copy() #현재 프레임 이미지 이전 프레임으로 재 저장

    for key in range(1, len(ddd) + 5):
        if key in ddd.keys():
            ddd_num[key][1] += 1
            if ddd_num[key][1] > 10: #10번이상 탬플릿 매칭 안될 시 삭제
                del ddd_num[key]
                del ddd[key]
            elif((ddd_num[key][1] > 4) and (ddd_num[key][0] < 10)):
                del ddd_num[key]
                del ddd[key]

    if select == 0:    # ROI 지역 접근시 파라미터 +1
        for key, value in ddd.items():
            x, y, w, h = value
            person_x = int(x + w / 2)
            person_y = int(y + h)
            if roi_img[person_y, person_x] == 255: ## 흰색 범위 배회
                ddd_num[key][2] += 1
            if roi_img[person_y, person_x] == 0:  ## 검정색 범위 침입
                ddd_num[key][3] += 1
                invade += 1

    if invade > 0:          #침입 시 깜박이
        if invade < 30:
            cv2.putText(image, 'WARNING! INVADE', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            invade += 1
        elif invade < 60:
            invade += 1
        elif invade < 90:
            invade = 1

    return image



def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious



def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)



