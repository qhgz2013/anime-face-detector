
def compat_import():
    import tensorflow as tf
    try:
        tf_v1 = tf.compat.v1
        tf_v1.disable_v2_behavior()
        return tf_v1
    except ImportError:
        return tf

compat_tensorflow = compat_import()
