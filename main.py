import numpy as np
import cv2
from faster_rcnn_wrapper import FasterRCNNSlim
import tensorflow as tf
import argparse
import os
import json


def nms(boxes, scores, thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    num_boxes = boxes.shape[0]
    suppressed = np.zeros([num_boxes], dtype=np.int32)

    keep = []
    for _i in range(num_boxes):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, num_boxes):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep


def detect(sess, rcnn_cls, image):
    img_origin = image.astype(np.float32, copy=True)
    img_origin -= np.array([[[102.9801, 115.9465, 112.7717]]])

    img_shape = img_origin.shape
    img_size_min = np.min(img_shape[:2])
    img_size_max = np.max(img_shape[:2])

    img_scale = 600 / img_size_min
    if np.round(img_scale * img_size_max) > 1000:
        img_scale = 1000 / img_size_max
    img = cv2.resize(img_origin, None, None, img_scale, img_scale, cv2.INTER_LINEAR)
    img_info = np.array([img.shape[0], img.shape[1], img_scale], dtype=np.float32)
    img = np.expand_dims(img, 0)
    _, scores, bbox_pred, rois = rcnn_cls.test_image(sess, img, img_info)

    boxes = rois[:, 1:] / img_scale

    boxes = boxes.astype(bbox_pred.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1
    heights = boxes[:, 3] - boxes[:, 1] + 1
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    dx = bbox_pred[:, 0::4]
    dy = bbox_pred[:, 1::4]
    dw = bbox_pred[:, 2::4]
    dh = bbox_pred[:, 3::4]
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_boxes = np.zeros_like(bbox_pred, dtype=bbox_pred.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    pred_boxes[:, 0::4] = np.maximum(pred_boxes[:, 0::4], 0)
    pred_boxes[:, 1::4] = np.maximum(pred_boxes[:, 1::4], 0)
    pred_boxes[:, 2::4] = np.minimum(pred_boxes[:, 2::4], img_shape[1] - 1)
    pred_boxes[:, 3::4] = np.minimum(pred_boxes[:, 3::4], img_shape[0] - 1)
    return scores, pred_boxes


def load_file_from_dir(dir_path):
    ret = []
    for file in os.listdir(dir_path):
        path_comb = os.path.join(dir_path, file)
        if os.path.isdir(path_comb):
            ret += load_file_from_dir(path_comb)
        else:
            ret.append(path_comb)
    return ret


def main():
    parser = argparse.ArgumentParser(description='Anime face detector demo')
    parser.add_argument('-i', help='The input path of an image or directory', required=True, dest='input', type=str)
    parser.add_argument('-o', help='The output json path of the detection result', dest='output')
    parser.add_argument('-nms', help='Change the threshold for non maximum suppression',
                        dest='nms_thresh', default=0.3, type=float)
    parser.add_argument('-conf', help='Change the threshold for class regression', dest='conf_thresh',
                        default=0.8, type=float)
    parser.add_argument('-model', help='Specify a new path for model', dest='model', type=str,
                        default='model/res101_faster_rcnn_iter_60000.ckpt')

    args = parser.parse_args()

    assert os.path.exists(args.input), 'The input path does not exists'

    if os.path.isdir(args.input):
        files = load_file_from_dir(args.input)
    else:
        files = [args.input]

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)

    net = FasterRCNNSlim()
    saver = tf.train.Saver()

    saver.restore(sess, args.model)

    result = {}

    for file in files:
        img = cv2.imread(file)
        scores, boxes = detect(sess, net, img)
        boxes = boxes[:, 4:8]
        scores = scores[:, 1]
        keep = nms(boxes, scores, args.nms_thresh)
        boxes = boxes[keep, :]
        scores = scores[keep]
        inds = np.where(scores >= args.conf_thresh)[0]
        scores = scores[inds]
        boxes = boxes[inds, :]

        result[file] = {}
        result[file]['bbox'] = []
        for i in range(scores.shape[0]):
            result[file]['score'] = float(scores[i])
            x1, y1, x2, y2 = boxes[i, :].tolist()
            result[file]['bbox'].append([x1, y1, x2, y2])

            if args.output is None:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        if args.output is None:
            cv2.imshow(file, img)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f)
    else:
        cv2.waitKey()


if __name__ == '__main__':
    main()
