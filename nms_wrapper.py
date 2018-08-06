from enum import Enum


class NMSType(Enum):
    PY_NMS = 1
    CPU_NMS = 2
    GPU_NMS = 3


default_nms_type = NMSType.PY_NMS


class NMSWrapper:
    def __init__(self, nms_type=default_nms_type):
        assert type(nms_type) == NMSType
        if nms_type == NMSType.PY_NMS:
            from nms.py_cpu_nms import py_cpu_nms
            self._nms = py_cpu_nms
        elif nms_type == NMSType.CPU_NMS:
            from nms.cpu_nms import cpu_nms
            self._nms = cpu_nms
        elif nms_type == NMSType.GPU_NMS:
            from nms.gpu_nms import gpu_nms
            self._nms = gpu_nms
        else:
            raise ValueError('current nms type is not implemented yet')

    def __call__(self, *args, **kwargs):
        return self._nms(*args, **kwargs)
