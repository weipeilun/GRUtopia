from grutopia.core.constants import DEFAULT_CAMERA_RESOLUTION


def get_camera_resolution(config):
    if config.resolution_x is None or config.resolution_y is None:
        return DEFAULT_CAMERA_RESOLUTION
    else:
        return (config.resolution_y, config.resolution_x, 3)