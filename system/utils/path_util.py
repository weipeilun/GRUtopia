import sys
import os


def get_relative_path(path, base_path=None):
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), path)
    else:
        if not base_path:
            return path
        else:
            return os.path.join(base_path, path)
