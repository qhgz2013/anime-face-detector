import numpy as np
import cv2
from faster_rcnn_wrapper import FasterRCNNSlim
import tensorflow as tf
import argparse
import os
import json
import time
from nms_wrapper import NMSType, NMSWrapper


def detect(sess, rcnn_cls, image):
    # pre-processing image for Faster-RCNN
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

    # test image
    _, scores, bbox_pred, rois = rcnn_cls.test_image(sess, img, img_info)

    # bbox transform
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
    # clipping edge
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


def fmt_time(dtime):
    if dtime <= 0:
        return '0:00.000'
    elif dtime < 60:
        return '0:%02d.%03d' % (int(dtime), int(dtime * 1000) % 1000)
    elif dtime < 3600:
        return '%d:%02d.%03d' % (int(dtime / 60), int(dtime) % 60, int(dtime * 1000) % 1000)
    else:
        return '%d:%02d:%02d.%03d' % (int(dtime / 3600), int((dtime % 3600) / 60), int(dtime) % 60,
                                      int(dtime * 1000) % 1000)


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
    parser.add_argument('-nms-type', help='Type of nms', choices=['PY_NMS', 'CPU_NMS', 'GPU_NMS'], dest='nms_type',
                        default='CPU_NMS')

    args = parser.parse_args()

    assert os.path.exists(args.input), 'The input path does not exists'

    if os.path.isdir(args.input):
        files = load_file_from_dir(args.input)
    else:
        files = [args.input]
    file_len = len(files)

    if args.nms_type == 'PY_NMS':
        nms_type = NMSType.PY_NMS
    elif args.nms_type == 'CPU_NMS':
        nms_type = NMSType.CPU_NMS
    elif args.nms_type == 'GPU_NMS':
        nms_type = NMSType.GPU_NMS
    else:
        raise ValueError('Incorrect NMS Type, not supported yet')

    nms = NMSWrapper(nms_type)

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)

    net = FasterRCNNSlim()
    saver = tf.train.Saver()

    saver.restore(sess, args.model)

    result = {}

    time_start = time.time()

    for idx, file in enumerate(files):
        elapsed = time.time() - time_start
        eta = (file_len - idx) * elapsed / idx if idx > 0 else 0
        print('[%d/%d] Elapsed: %s, ETA: %s >> %s' % (idx+1, file_len, fmt_time(elapsed), fmt_time(eta), file))
        img = cv2.imread(file)
        scores, boxes = detect(sess, net, img)
        boxes = boxes[:, 4:8]
        scores = scores[:, 1]
        keep = nms(np.hstack([boxes, scores[:, np.newaxis]]).astype(np.float32), args.nms_thresh)
        boxes = boxes[keep, :]
        scores = scores[keep]
        inds = np.where(scores >= args.conf_thresh)[0]
        scores = scores[inds]
        boxes = boxes[inds, :]

        result[file] = []
        for i in range(scores.shape[0]):
            x1, y1, x2, y2 = boxes[i, :].tolist()
            new_result = {'score': float(scores[i]),
                          'bbox': [x1, y1, x2, y2]}
            result[file].append(new_result)

            if args.output is None:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        if args.output:
            if ((idx+1) % 1000) == 0:
                # saving the temporary result
                with open(args.output, 'w') as f:
                    json.dump(result, f)
        else:
            cv2.imshow(file, img)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f)
    else:
        cv2.waitKey()


if __name__ == '__main__':
    main()
