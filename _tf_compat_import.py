__all__ = ['compat_tensorflow']

def _compat_tf_import(enable_gpu: bool = True):
    if not enable_gpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    try:
        tf_v1 = tf.compat.v1
        tf_v1.disable_v2_behavior()
        return tf_v1
    except ImportError:
        return tf

compat_tensorflow = _compat_tf_import()
