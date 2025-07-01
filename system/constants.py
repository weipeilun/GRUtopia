# -*- coding: utf-8-*-


CONFIG_KEY_FREQUENCY = 'frequency'
CONFIG_KEY_OUTPUT_FREQUENCY = 'output_frequency'
CONFIG_KEY_IN_SOURCE = 'in_source'
CONFIG_KEY_IN_IP = 'in_ip'
CONFIG_KEY_OUT_PORT = 'out_port'
CONFIG_KEY_SIGNAL_OUT_PORT = 'signal_out_port'
CONFIG_KEY_RESIZE_HEIGHT = 'resize_height'
CONFIG_KEY_RESIZE_WIDTH = 'resize_width'
CONFIG_KEY_VISUALIZATION_HEIGHT = 'visualization_height'
CONFIG_KEY_VISUALIZATION_WIDTH = 'visualization_width'
CONFIG_KEY_BODY_BBOX_VALID_PROPORTION = 'body_bbox_valid_proportion'
CONFIG_KEY_WATCH_FREQUENCY = 'watch_frequency'
CONFIG_KEY_WHEEL_FREQUENCY = 'wheel_frequency'
CONFIG_KEY_KEEP_SECONDS = 'keep_seconds'
CONFIG_KEY_MILLIMETER_WAVE_RADAR_FREQUENCY = 'millimeter_wave_radar_frequency'
CONFIG_KEY_GAZE_DO_VISUALIZATION = 'gaze_do_visualization'
CONFIG_KEY_GAZE_VISUALIZATION_LENGTH = 'gaze_visualization_length'
CONFIG_KEY_HEAD_POSE_VISUALIZATION_LENGTH = 'head_pose_visualization_length'
CONFIG_KEY_OBJECT_DO_VISUALIZATION = 'object_do_visualization'
CONFIG_KEY_AVATAR_DO_VISUALIZATION = 'avatar_do_visualization'
CONFIG_KEY_AVATAR_RTMP_URL = 'avatar_rtmp_url'
CONFIG_KEY_MIN_VALID_FRAMES = 'min_valid_frames'
CONFIG_KEY_MODEL = 'model'
CONFIG_KEY_SERVER_IP = 'server_ip'
CONFIG_KEY_OUTPUT_MASK = 'output_mask'
CONFIG_KEY_IN_QUEUE = 'in_queue'
CONFIG_KEY_WAVE_FREQUENCY = 'wave_frequency'
CONFIG_KEY_CAMERA_PORT_ID = 'camera_port_id'
CONFIG_KEY_CAMERA_POSITIONS = 'camera_positions'
CONFIG_KEY_CAMERA_ATTRIBUTES = 'camera_attributes'
CONFIG_KEY_PITCH = 'pitch'
CONFIG_KEY_YAW = 'yaw'
CONFIG_KEY_CAMERA_CALIBRATION = 'calibration'
CONFIG_KEY_CAMERA_GROUP = 'camera_group'
CONFIG_KEY_VALID_CAMERAS = 'valid_cameras'
CONFIG_KEY_VALID_WIDE_CAMERAS = 'valid_wide_cameras'
CONFIG_KEY_VALID_IR_CAMERAS = 'valid_ir_cameras'
CONFIG_KEY_TORCH_DEVICE = 'torch_device'
CONFIG_KEY_TENSORFLOW_DEVICE = 'tensorflow_device'
CONFIG_KEY_PADDLE_DEVICE = 'paddle_device'
CONFIG_KEY_ONNX_DEVICE = 'onnx_device'
CONFIG_KEY_OUT_QUEUE_SHAPE = 'out_queue_shape'
CONFIG_KEY_OUT_QUEUE_DTYPE = 'out_queue_dtype'
CONFIG_KEY_VERSION = 'version'
CONFIG_KEY_STREAM_FREQUENCY = 'stream_frequency'
CONFIG_KEY_TASK_CONFIG_FILE_PATH = 'task_config_file_path'
CONFIG_KEY_ROBOT_CONFIG_FILE_PATH = 'robot_config_file_path'
CONFIG_KEY_OBSERVATION_KEY_LIST = 'observation_key_list'
CONFIG_KEY_ROBOT_OUTPUT_CAMERAS = 'output_cameras'

CONFIG_KEY_DEFAULT_TORCH_DEVICE = 'default_torch_device'
CONFIG_KEY_DEFAULT_TENSORFLOW_DEVICE = 'default_tensorflow_device'
CONFIG_KEY_DEFAULT_PADDLE_DEVICE = 'default_paddle_device'
CONFIG_KEY_DEFAULT_ONNX_DEVICE = 'default_onnx_device'

CONFIG_KEY_ISAAC_SIM = 'isaac_sim'
CONFIG_KEY_VIDEO_SERVER = 'video_server'
CONFIG_KEY_TEST = 'test'
CONFIG_KEY_MOCK = 'mock'

SIGNAL_KEY_IS_VALID = 'is_valid'
SIGNAL_KEY_STARTED = 'started'

SIGNAL_KEY_CAMERA_POSITION = 'camera_position'
SIGNAL_KEY_CAMERA_NUM = 'camera_num'

SIGNAL_KEY_MICROPHONE_CHANNEL = 'microphone_channel'

SIGNAL_KEY_BODY_BBOX = 'body_bbox'
SIGNAL_KEY_BODY_TRACKING_ID = 'body_tracking_id'
SIGNAL_KEY_FACE_BBOX = 'face_bbox'
SIGNAL_KEY_FACE_LANDMARK = 'face_landmark'

SIGNAL_KEY_SUPPLIER_CHOICE = 'supplier_choice'

SIGNAL_KEY_BPM = 'bpm'
SIGNAL_KEY_HRV_SDNN = 'hrv_sdnn'
SIGNAL_KEY_HRV_LF_HF = 'hrv_lf_hf'
SIGNAL_KEY_HRV_STRESS_LEVEL = 'hrv_stress_level'
SIGNAL_KEY_BPM_WAVE = 'bpm_wave'
SIGNAL_KEY_BPM_WAVE_VISUALIZATION = 'bpm_wave_visualization'
SIGNAL_KEY_BPM_TIEJIANGJUN = 'bpm_tiejiangjun'
SIGNAL_KEY_BPM_NURALOGIX = 'bpm_nuralogix'
SIGNAL_KEY_BPM_BEYONCA = 'bpm_beyonca'

SIGNAL_KEY_BLOOD_PRESSURE_HIGH = 'blood_pressure_high'
SIGNAL_KEY_BLOOD_PRESSURE_HIGH_TIEJIANGJUN = 'blood_pressure_high_tiejiangjun'
SIGNAL_KEY_BLOOD_PRESSURE_HIGH_NURALOGIX = 'blood_pressure_high_nuralogix'
SIGNAL_KEY_BLOOD_PRESSURE_HIGH_BEYONCA = 'blood_pressure_high_beyonca'
SIGNAL_KEY_BLOOD_PRESSURE_LOW = 'blood_pressure_low'
SIGNAL_KEY_BLOOD_PRESSURE_LOW_TIEJIANGJUN = 'blood_pressure_low_tiejiangjun'
SIGNAL_KEY_BLOOD_PRESSURE_LOW_NURALOGIX = 'blood_pressure_low_nuralogix'
SIGNAL_KEY_BLOOD_PRESSURE_LOW_BEYONCA = 'blood_pressure_low_beyonca'

SIGNAL_KEY_BODY_TEMPERATURE = 'body_temperature'
SIGNAL_KEY_BODY_TEMPERATURE_WAVE = 'body_temperature_wave'
SIGNAL_KEY_BODY_TEMPERATURE_IMAGE = 'body_temperature_image'
# 目前等价于driver_status->CONSCIOUSNESS
SIGNAL_KEY_FATIGUE = 'fatigue'
SIGNAL_KEY_CRITICAL_ILLNESS = 'critical_illness'

SIGNAL_KEY_DMS_AWARENESS = 'dms_awareness'
SIGNAL_KEY_DMS_CONSCIOUSNESS = 'dms_consciousness'
SIGNAL_KEY_DMS_CRITICAL_ILLNESS = 'dms_critical_illness'
SIGNAL_KEY_DMS_SIT_STRAIGHT = 'dms_sit_straight'
SIGNAL_KEY_DMS_BAD_POSE = 'dms_bad_pose'
SIGNAL_KEY_DMS_NUM_BAD_EYES = 'dms_num_bad_eyes'
SIGNAL_KEY_DMS_NUM_BAD_POSE = 'dms_num_bad_pose'
SIGNAL_KEY_DMS_BAD_EYES_FILTER = 'dms_bad_eyes_filter'
SIGNAL_KEY_DMS_BAD_POSE_FILTER = 'dms_bad_pose_filter'
SIGNAL_KEY_DMS_DISTRACTED_FILTER = 'dms_distracted_filter'
SIGNAL_KEY_DMS_UNCONSCIOUS_FILTER = 'dms_unconscious_filter'
SIGNAL_KEY_DMS_CRITICAL_ILLNESS_FILTER = 'dms_critical_illness_filter'
SIGNAL_KEY_DMS_SIT_STRAIGHT_FILTER = 'dms_sit_straight_filter'
SIGNAL_KEY_DMS_PITCH_MEAN = 'dms_pitch_mean'
SIGNAL_KEY_DMS_YAW_MEAN = 'dms_yaw_mean'
SIGNAL_KEY_DMS_NUM_DISTRACTED = 'dms_num_distracted'
SIGNAL_KEY_DMS_IS_SIGHT_BLOCKED = 'is_sight_blocked'
SIGNAL_KEY_DMS_IS_EYE_CLOSED = 'is_eye_closed'
SIGNAL_KEY_DMS_IS_HEAD_NOD = 'is_head_nod'
SIGNAL_KEY_DMS_IS_HEAD_SHAKE = 'is_head_shake'

SIGNAL_KEY_BODY_SPO2 = 'body_spo2'
SIGNAL_KEY_BODY_SPO2_TIEJIANGJUN = 'body_spo2_tiejiangjun'
SIGNAL_KEY_BODY_SPO2_NURALOGIX = 'body_spo2_nuralogix'
SIGNAL_KEY_BODY_SPO2_BEYONCA = 'body_spo2_beyonca'

SIGNAL_KEY_ECG = 'ecg'
SIGNAL_KEY_PPG = 'ppg'

SIGNAL_KEY_RESET_UNCONSCIOUSNESS = 'reset_unconsciousness'
SIGNAL_KEY_FORCE_HEART_ATTACK = 'force_heart_attack'

SIGNAL_KEY_EMOTION = 'emotion'
SIGNAL_KEY_EMOTION_DISTRIBUTION = 'emotion_distribution'
SIGNAL_KEY_EMOTION_LEVEL = 'emotion_level'

SIGNAL_KEY_AGE = 'age'

SIGNAL_KEY_SEG_MASK = 'seg_mask'

SIGNAL_KEY_RECOLOR = 'recolor'

SIGNAL_KEY_HAND_GESTURE = 'hand_gesture'

SIGNAL_KEY_OBJECTS = 'objects'

SIGNAL_KEY_FRAME_ID = 'frame_id'
SIGNAL_KEY_IMAGE_POSITION_LIST = 'image_position_list'

SIGNAL_KEY_GAZE_VISUALIZATION = 'gaze_visualization'
SIGNAL_KEY_GAZE_CENTER_VISUALIZATION = 'gaze_center_visualization'
SIGNAL_KEY_GAZE_INFO = 'gaze_info'
# SIGNAL_KEY_BBOX = 'bbox'
SIGNAL_KEY_GAZE_CENTER = 'gaze_center'
SIGNAL_KEY_GAZE_VECTOR = 'gaze_vector'
SIGNAL_KEY_HEAD_ROT_MAT = 'head_rot'
SIGNAL_KEY_HEAD_TRANS_MAT = 'head_trans'
SIGNAL_KEY_GAZE_CENTER_CALIBRATED = 'gaze_center_calibrated'
SIGNAL_KEY_GAZE_VECTOR_CALIBRATED = 'gaze_vector_calibrated'
SIGNAL_KEY_HEAD_ROT_MAT_CALIBRATED = 'head_rot_calibrated'
SIGNAL_KEY_HEAD_TRANS_MAT_CALIBRATED = 'head_trans_calibrated'

SIGNAL_KEY_GAZE_TRACKING_INFO = 'gaze_tracking_info'
SIGNAL_KEY_GAZE_HEAD_ROT_Y_EULER = 'gaze_tracking_head_rot_y_euler'
SIGNAL_KEY_GAZE_VECTOR_YZ = 'gaze_vector_yz'
SIGNAL_KEY_GAZE_VECTOR_YZ_PROJECTION_LENGTH_FACTOR = 'gaze_vector_yz_projection_length_factor'
SIGNAL_KEY_GAZE_TRACKING_HEAD_CENTER = 'gaze_tracking_head_center'
SIGNAL_KEY_GAZE_TRACKING_HEAD_RADIUS = 'gaze_tracking_head_radius'
SIGNAL_KEY_GAZE_VECTOR_YZ_START_COORDINATES = 'gaze_vector_yz_start_coordinates'
SIGNAL_KEY_GAZE_VECTOR_YZ_END_COORDINATES = 'gaze_vector_yz_end_coordinates'

SIGNAL_KEY_GAZE_FOCUS_COORDINATES = 'gaze_focus_coordinates'
SIGNAL_KEY_GAZE_FOCUS_COORDINATES_FRONT = 'gaze_focus_coordinates_front'
SIGNAL_KEY_GAZE_FOCUS_COORDINATES_MIDDLE = 'gaze_focus_coordinates_middle'
SIGNAL_KEY_GAZE_FOCUS_COORDINATES_BACK = 'gaze_focus_coordinates_back'

SIGNAL_KEY_SUMMARY_EVENT_ID = 'event_id'
SIGNAL_KEY_SUMMARY_EVENT_TYPE = 'event_type'

SIGNAL_KEY_FACE_ID = 'face_id'
SIGNAL_KEY_ROLE_IS_PRESENT = 'is_present'
SIGNAL_KEY_USER_ID = 'user_id'
SIGNAL_KEY_USER_IS_REGISTERED = 'user_registered'
SIGNAL_KEY_USER_PICTURE = 'user_picture'
SIGNAL_KEY_USER_BACKGROUND = 'user_background'
SIGNAL_KEY_USER_NAME = 'user_name'
SIGNAL_KEY_USER_COLOR = 'user_color'
SIGNAL_KEY_USER_AGE = 'user_age'
SIGNAL_KEY_USER_GENDER = 'user_gender'
SIGNAL_KEY_USER_RACE = 'user_race'
SIGNAL_KEY_USER_FACE_EMBEDDING = 'user_face_embedding'

SIGNAL_KEY_COUGH_SNEEZE = 'cough_sneeze'

SIGNAL_KEY_DEFAULT_YOLO_INFO = 'yolo_info'
SIGNAL_KEY_FASHION_INFO = 'fashion_info'
SIGNAL_KEY_SMOKE_INFO = 'smoke_info'

SIGNAL_KEY_BBOX = 'bbox'
SIGNAL_KEY_CLASS_ID = 'class_id'
SIGNAL_KEY_SCORE = 'score'
SIGNAL_KEY_MASK = 'mask'
SIGNAL_KEY_IMAGE = 'image'
SIGNAL_KEY_ATTRIBUTES = 'attributes'

SIGNAL_KEY_ILLNESS_BODY_TEMPERATURE = 'illness_body_temperature'
SIGNAL_KEY_ILLNESS_HEART_RATE = 'illness_heart_rate'
SIGNAL_KEY_ILLNESS_BREATHE = 'illness_breathe'
SIGNAL_KEY_ILLNESS_OVERALL = 'illness_overall'

SIGNAL_KEY_YAWN = 'yawn'

SIGNAL_KEY_HEALTH_STATUS = 'health_status'
SIGNAL_KEY_HEART_RATE = 'heart_rate'
SIGNAL_KEY_BREATHE_RATE = 'breathe_rate'
SIGNAL_KEY_HEART_RATE_LEVEL = 'heart_rate_level'
SIGNAL_KEY_BREATHE_RATE_LEVEL = 'breathe_rate_level'
SIGNAL_KEY_BODY_TEMPERATURE_LEVEL = 'body_temperature_level'
SIGNAL_KEY_BLOOD_PRESSURE_LOW_LEVEL = 'blood_pressure_low_level'
SIGNAL_KEY_BLOOD_PRESSURE_HIGH_LEVEL = 'blood_pressure_high_level'
SIGNAL_KEY_CRITICAL_ILLNESS_LEVEL = 'critical_illness_level'

SIGNAL_KEY_HEART_RATE_SUPPLIER = 'heart_rate_supplier'
SIGNAL_KEY_HRV_SUPPLIER = 'hrv_supplier'
SIGNAL_KEY_BREATHE_SUPPLIER = 'breathe_supplier'
SIGNAL_KEY_BLOOD_PRESSURE_SUPPLIER = 'blood_pressure_supplier'
SIGNAL_KEY_BODY_TEMPERATURE_SUPPLIER = 'body_temperature_supplier'
SIGNAL_KEY_BODY_SPO2_SUPPLIER = 'body_spo2_supplier'

SIGNAL_KEY_ECG_WHEEL = 'ecg_wheel'
SIGNAL_KEY_ECG_WHEEL_ECG_VALID = 'ecg_wheel_ecg_valid'
SIGNAL_KEY_ECG_WHEEL_ILLNESS = 'ecg_wheel_illness'
SIGNAL_KEY_ECG_WHEEL_HEART_RATE = 'ecg_wheel_heart_rate'
SIGNAL_KEY_ECG_WHEEL_BREATH = 'ecg_wheel_breath'

SIGNAL_KEY_OBJECTS_ATTRIBUTE_LIVING_THINGS = 'living_things'

SIGNAL_KEY_MOTION_ID = 'motion_id'

SIGNAL_KEY_GLASSES = 'glasses'
SIGNAL_KEY_HAT = 'hat'
SIGNAL_KEY_EARRING = 'earring'
SIGNAL_KEY_NECKLACE = 'necklace'
SIGNAL_KEY_NECKTIE = 'necktie'

SIGNAL_KEY_TIME = 'time'

FACE_ATTRIBUTE_KEY_GENDER = 'gender'
FACE_ATTRIBUTE_KEY_AGE = 'age'
FACE_ATTRIBUTE_KEY_RACE = 'race'

# ADI(CBU) watch data type
CBU_TYPE_UNKNOWN = 0
CBU_TYPE_ECG = 1
CBU_TYPE_PPG = 2
CBU_TYPE_ECG_PPG = 3

# Microphone
# MICROPHONE_RATE = 44100
# MICROPHONE_CHANNELS = 1
MICROPHONE_RATE = 16000
MICROPHONE_CHANNELS = 2
