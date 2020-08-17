from _tf_compat_import import compat_tensorflow as tf
from tf_contrib.resnet_v1 import resnet_v1_block, resnet_v1
import tf_contrib.slim as slim
from tf_contrib.resnet_utils import arg_scope, conv2d_same
import numpy as np


class FasterRCNNSlim:

    def __init__(self):
        self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                        resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                        resnet_v1_block('block3', base_depth=256, num_units=23, stride=1),
                        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])

        self._anchor_scales = [4, 8, 16, 32]
        self._num_scales = len(self._anchor_scales)

        self._anchor_ratios = [1]
        self._num_ratios = len(self._anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios
        self._scope = 'resnet_v1_101'

        with arg_scope([slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.separable_conv2d,
                        slim.fully_connected],
                       weights_regularizer=slim.l2_regularizer(0.0001),
                       biases_regularizer=tf.no_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            # in _build_network
            initializer = tf.random_normal_initializer(stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(stddev=0.001)
            # in _image_to_head
            with slim.arg_scope(self._resnet_arg_scope()):
                # in _build_base
                with tf.variable_scope(self._scope, self._scope):
                    net_conv = conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
                    net_conv = tf.pad(net_conv, [[0, 0], [1, 1], [1, 1], [0, 0]])
                    net_conv = slim.max_pool2d(net_conv, [3, 3], stride=2, padding='VALID', scope='pool1')
                net_conv, _ = resnet_v1(net_conv, self._blocks[:-1], global_pool=False, include_root_block=False,
                                        scope=self._scope)
            with tf.variable_scope(self._scope, self._scope):
                # in _anchor_component
                with tf.variable_scope('ANCHOR-default'):
                    height = tf.cast(tf.ceil(self._im_info[0] / 16.0), dtype=tf.int32)
                    width = tf.cast(tf.ceil(self._im_info[1] / 16.0), dtype=tf.int32)

                    shift_x = tf.range(width) * 16
                    shift_y = tf.range(height) * 16
                    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
                    sx = tf.reshape(shift_x, [-1])
                    sy = tf.reshape(shift_y, [-1])
                    shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
                    k = width * height
                    shifts = tf.transpose(tf.reshape(shifts, [1, k, 4]), perm=[1, 0, 2])

                    anchors = np.array([[-24, -24, 39, 39], [-56, -56, 71, 71],
                                        [-120, -120, 135, 135], [-248, -248, 263, 263]], dtype=np.int32)

                    a = anchors.shape[0]
                    anchor_constant = tf.constant(anchors.reshape([1, a, 4]), dtype=tf.int32)
                    length = k * a
                    anchors_tf = tf.reshape(anchor_constant + shifts, shape=[length, 4])
                    anchors = tf.cast(anchors_tf, dtype=tf.float32)
                    self._anchors = anchors
                    self._anchor_length = length

                # in _region_proposal
                rpn = slim.conv2d(net_conv, 512, [3, 3], trainable=False, weights_initializer=initializer,
                                  scope='rpn_conv/3x3')
                rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=False,
                                            weights_initializer=initializer, padding='VALID', activation_fn=None,
                                            scope='rpn_cls_score')
                rpn_cls_score_reshape = self._reshape(rpn_cls_score, 2, 'rpn_cls_score_reshape')
                rpn_cls_prob_reshape = self._softmax(rpn_cls_score_reshape, 'rpn_cls_prob_reshape')
                # rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name='rpn_cls_pred')
                rpn_cls_prob = self._reshape(rpn_cls_prob_reshape, self._num_anchors * 2, 'rpn_cls_prob')
                rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=False,
                                            weights_initializer=initializer, padding='VALID', activation_fn=None,
                                            scope='rpn_bbox_pred')

                # in _proposal_layer
                with tf.variable_scope('rois'):
                    post_nms_topn = 300
                    nms_thresh = 0.7
                    scores = rpn_cls_prob[:, :, :, self._num_anchors:]
                    scores = tf.reshape(scores, [-1])
                    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

                    boxes = tf.cast(self._anchors, rpn_bbox_pred.dtype)
                    widths = boxes[:, 2] - boxes[:, 0] + 1.0
                    heights = boxes[:, 3] - boxes[:, 1] + 1.0
                    ctr_x = boxes[:, 0] + widths * 0.5
                    ctr_y = boxes[:, 1] + heights * 0.5

                    dx = rpn_bbox_pred[:, 0]
                    dy = rpn_bbox_pred[:, 1]
                    dw = rpn_bbox_pred[:, 2]
                    dh = rpn_bbox_pred[:, 3]

                    pred_ctr_x = dx * widths + ctr_x
                    pred_ctr_y = dy * heights + ctr_y
                    pred_w = tf.exp(dw) * widths
                    pred_h = tf.exp(dh) * heights

                    pred_boxes0 = pred_ctr_x - pred_w * 0.5
                    pred_boxes1 = pred_ctr_y - pred_h * 0.5
                    pred_boxes2 = pred_ctr_x + pred_w * 0.5
                    pred_boxes3 = pred_ctr_y + pred_h * 0.5

                    b0 = tf.clip_by_value(pred_boxes0, 0, self._im_info[1] - 1)
                    b1 = tf.clip_by_value(pred_boxes1, 0, self._im_info[0] - 1)
                    b2 = tf.clip_by_value(pred_boxes2, 0, self._im_info[1] - 1)
                    b3 = tf.clip_by_value(pred_boxes3, 0, self._im_info[0] - 1)

                    proposals = tf.stack([b0, b1, b2, b3], axis=1)
                    indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topn,
                                                           iou_threshold=nms_thresh)
                    boxes = tf.cast(tf.gather(proposals, indices), dtype=tf.float32)
                    # rpn_scores = tf.reshape(tf.gather(scores, indices), [-1, 1])

                    batch_inds = tf.zeros([tf.shape(indices)[0], 1], dtype=tf.float32)
                    rois = tf.concat([batch_inds, boxes], 1)

                # in _crop_pool_layer
                with tf.variable_scope('pool5'):
                    batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name='bath_id'), [1])
                    bottom_shape = tf.shape(net_conv)
                    height = (tf.cast(bottom_shape[1], dtype=tf.float32) - 1) * 16.0
                    width = (tf.cast(bottom_shape[2], dtype=tf.float32) - 1) * 16.0
                    x1 = tf.slice(rois, [0, 1], [-1, 1], name='x1') / width
                    y1 = tf.slice(rois, [0, 2], [-1, 1], name='y1') / height
                    x2 = tf.slice(rois, [0, 3], [-1, 1], name='x2') / width
                    y2 = tf.slice(rois, [0, 4], [-1, 1], name='y2') / height
                    bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
                    pool5 = tf.image.crop_and_resize(net_conv, bboxes, tf.cast(batch_ids, dtype=tf.int32), [7, 7], 
                                                     name='crops')
            # in _head_to_tail
            with slim.arg_scope(self._resnet_arg_scope()):
                fc7, _ = resnet_v1(pool5, self._blocks[-1:], global_pool=False, include_root_block=False,
                                   scope=self._scope)
                fc7 = tf.reduce_mean(fc7, axis=[1, 2])
            with tf.variable_scope(self._scope, self._scope):
                # in _region_classification
                cls_score = slim.fully_connected(fc7, 2, weights_initializer=initializer, trainable=False,
                                                 activation_fn=None, scope='cls_score')
                cls_prob = self._softmax(cls_score, 'cls_prob')
                # cls_pred = tf.argmax(cls_score, 'cls_pred')
                bbox_pred = slim.fully_connected(fc7, 2*4, weights_initializer=initializer_bbox, trainable=False,
                                                 activation_fn=None, scope='bbox_pred')
        self._cls_score = cls_score
        self._cls_prob = cls_prob
        self._bbox_pred = bbox_pred
        self._rois = rois

        stds = np.tile(np.array([0.1, 0.1, 0.2, 0.2]), 2)
        means = np.tile(np.array([0.0, 0.0, 0.0, 0.0]), 2)
        self._bbox_pred *= stds
        self._bbox_pred += means

    @staticmethod
    def _resnet_arg_scope():
        batch_norm_params = {
            'is_training': False,
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'trainable': False,
            'updates_collections': tf.GraphKeys.UPDATE_OPS
        }
        with arg_scope([slim.conv2d],
                       weights_regularizer=slim.l2_regularizer(0.0001),
                       weights_initializer=slim.variance_scaling_initializer(),
                       trainable=False,
                       activation_fn=tf.nn.relu,
                       normalizer_fn=slim.batch_norm,
                       normalizer_params=batch_norm_params):
            with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
                return arg_sc

    @staticmethod
    def _reshape(bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name):
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            reshaped = tf.reshape(to_caffe, [1, num_dim, -1, input_shape[2]])
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
        return to_tf

    @staticmethod
    def _softmax(bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def test_image(self, sess, image, im_info):
        return sess.run([self._cls_score, self._cls_prob, self._bbox_pred, self._rois], feed_dict={
            self._image: image,
            self._im_info: im_info
        })
