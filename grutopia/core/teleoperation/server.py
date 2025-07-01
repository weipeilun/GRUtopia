import multiprocessing
from multiprocessing import shared_memory
import queue
import numpy as np
from system.tools import interrupt


class TeleoperateServer(multiprocessing.Process):
    def __init__(self, left_shm_name, right_shm_name, left_eye_shape, right_eye_shape, data_dtype):
        super().__init__()
        self.left_shm_name = left_shm_name
        self.right_shm_name = right_shm_name
        self.left_eye_shape = left_eye_shape
        self.right_eye_shape = right_eye_shape
        self.data_dtype = data_dtype
        self.teleoperate_trigger_queue = multiprocessing.Queue()
        
        self.left_shm = None
        self.right_shm = None
        self.left_array = None
        self.right_array = None
        
        self.left_eye_buffer = []
        self.right_eye_buffer = []
        self.video_frame_count = 0
        
    def initialize_shared_memory(self):
        self.left_shm = shared_memory.SharedMemory(name=self.left_shm_name)
        self.right_shm = shared_memory.SharedMemory(name=self.right_shm_name)
        self.left_array = np.ndarray(self.left_eye_shape, dtype=self.dtype, buffer=self.left_shm.buf)
        self.right_array = np.ndarray(self.right_eye_shape, dtype=self.dtype, buffer=self.right_shm.buf)
        
    def process_frames(self):
        while True:
            if interrupt.interrupt_callback():
                break

            try:
                if self.video_frame_count > 300:
                    break
                
                self.teleoperate_trigger_queue.get(block=True, timeout=0.001)
                left_array_copy = np.copy(self.left_array)
                right_array_copy = np.copy(self.right_array)
                
                self.left_eye_buffer.append(left_array_copy)
                self.right_eye_buffer.append(right_array_copy)
                self.video_frame_count += 1
                print(self.video_frame_count)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in image processing: {str(e)}")
                continue
    
    def cleanup(self):
        if self.left_shm:
            self.left_shm.close()
        if self.right_shm:
            self.right_shm.close()
    
    def run(self):
        self.initialize_shared_memory()
        self.process_frames()
        self.cleanup()
