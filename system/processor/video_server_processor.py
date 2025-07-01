# encoding: utf-8
import traceback
import time
import logging
import numpy as np
from vuer import Vuer
from multiprocessing import Array, Value, shared_memory, Event
from vuer.schemas import ImageBackground, WebRTCStereoVideoPlane, DefaultScene
from system.constants import (
    CONFIG_KEY_VIDEO_SERVER,
    CONFIG_KEY_VALID_CAMERAS,
    CONFIG_KEY_ISAAC_SIM,
    CONFIG_KEY_IN_QUEUE,
)
from system.constants import CONFIG_KEY_ISAAC_SIM, CONFIG_KEY_VIDEO_SERVER, CONFIG_KEY_ROBOT_CONFIG_FILE_PATH, CONFIG_KEY_VALID_CAMERAS, CONFIG_KEY_OBSERVATION_KEY_LIST
from system.tools.processor_config import config
from system.processor.abstract_processor import AbstractProcessor
from system.constants import CONFIG_KEY_ISAAC_SIM
import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer
from av import VideoFrame
import json
import os
from system.processor.abstract_processor import AbstractProcessor
from system.tools.processor_config import config
import queue
import socket
import threading
from aiortc.rtcrtpsender import RTCRtpSender
from system.tools import interrupt

class SteroVideoTrack(MediaStreamTrack):
    kind = "video"
    
    def __init__(self, img_array, toggle_streaming, fps):
        super().__init__()  # Initialize base class
        # self.img_shape = (2*img_shape[0], img_shape[1], 3)
        # self.img_height, self.img_width = img_shape[:2]
        # self.shm_name = shm_name
        # existing_shm = shared_memory.SharedMemory(name=shm_name)
        # self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
        self.img_array = img_array
        self.toggle_streaming = toggle_streaming
        self.streaming_started = False
        self.timescale = 1000  # Use a timescale of 1000 for milliseconds
        self.frame_interval = 1 / fps
        self._last_frame_time = time.time()
        self.start_time = time.time()
    
    async def recv(self):
        """
        This method is called when a new frame is needed.
        """
        now = time.time()
        wait_time = self._last_frame_time + self.frame_interval - now
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        self._last_frame_time = time.time()
        # start = time.time()
        if not self.streaming_started:
            self.toggle_streaming.set()
            self.streaming_started = True
        frame = self.img_array.copy()
        # self.sem.release()
        # print("Time to get frame: ", time.time() - start, self.img_queue.qsize())
        # frame = self.img_array.copy()  # Assuming this is an async function to fetch a frame
        # frame = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        # print("recv")
        # start = time.time()
        av_frame = VideoFrame.from_ndarray(frame[0, 0], format='rgb24')  # Convert numpy array to AVFrame
        timestamp = int((time.time() - self.start_time) * self.timescale)
        av_frame.pts = timestamp
        av_frame.time_base = self.timescale
        # print("Time to process frame: ", time.time() - start)
        return av_frame
        

class VideoServerProcessor(AbstractProcessor):
    _PID = CONFIG_KEY_VIDEO_SERVER
    
    def __init__(self, in_queue_list=None, out_queue_info=None, interrupt_check=interrupt.interrupt_callback, listen_interval=0.01, do_validate=True, debug_time=False, log_level=logging.INFO, stream_mode="webrtc", cert_file="./cert.pem", key_file="./key.pem"):
        super().__init__(in_queue_list, out_queue_info, interrupt_check, listen_interval, do_validate, debug_time, log_level)
        self.stream_mode = stream_mode
        self.cert_file = cert_file
        self.key_file = key_file
        
        self.valid_camera_list = config[CONFIG_KEY_VALID_CAMERAS]
        self.img_shape = self.valid_camera_list[0][2][0]
        img_array_shape = (len(self.valid_camera_list), len(self.valid_camera_list[0][2]), *self.img_shape)
        self.img_array = np.zeros(img_array_shape, dtype=np.uint8)
        
        self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False))
        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if stream_mode == "image":
            self.app.spawn(start=False)(self.main_image)
        elif stream_mode == "webrtc":
            self.app.spawn(start=False)(self.main_webrtc)
        else:
            raise ValueError("stream_mode must be either 'webrtc' or 'image'")

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)
        if stream_mode == "webrtc":
            self.pcs = set()
        
            self.img_queue = queue.Queue()
            self.toggle_streaming = True
            self.fps = 30
            self.toggle_streaming = Event()

    async def start_webrtc_server(self):

        async def index(request):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            content = open(os.path.join(current_dir, "../resources/index3.html"), "r").read()
            return web.Response(content_type="text/html", text=content)

        async def javascript(request):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            content = open(os.path.join(current_dir, "../resources/client.js"), "r").read()
            return web.Response(content_type="application/javascript", text=content)

        def force_codec(pc, sender, forced_codec):
            kind = forced_codec.split("/")[0]
            codecs = RTCRtpSender.getCapabilities(kind).codecs
            transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
            transceiver.setCodecPreferences(
                [codec for codec in codecs if codec.mimeType == forced_codec]
            )
        
        async def offer(request):
            params = await request.json()
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

            pc = RTCPeerConnection()
            self.pcs.add(pc)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                print("Connection state is %s" % pc.connectionState)
                if pc.connectionState == "failed":
                    await pc.close()
                    self.pcs.discard(pc)

            # open media source
            zed_track = SteroVideoTrack(self.img_array, self.toggle_streaming, self.fps)
            video_sender = pc.addTrack(zed_track)
            # if Args.video_codec:
            force_codec(pc, video_sender, "video/H264")
            # elif Args.play_without_decoding:
            # raise Exception("You must specify the video codec using --video-codec")

            await pc.setRemoteDescription(offer)

            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
                ),
            )
            
        async def on_shutdown(app):
            # close peer connections
            coros = [pc.close() for pc in self.pcs]
            await asyncio.gather(*coros)
            self.pcs.clear()
            
        app = web.Application()
        # cors = aiohttp_cors.setup(app, defaults={
        #     "*": aiohttp_cors.ResourceOptions(
        #         allow_credentials=True,
        #         expose_headers="*",
        #         allow_headers="*",
        #         allow_methods="*",
        #     )
        # })
        # rtc = RTC((1024, 768), self.img_queue, toggle_streaming=True, fps=30, pcs=self.pcs)
        app.on_shutdown.append(on_shutdown)
        # cors.add(app.router.add_get("/", index))
        # cors.add(app.router.add_get("/client.js", javascript))
        # cors.add(app.router.add_post("/offer", offer))
        app.router.add_get("/", index)
        app.router.add_get("/client.js", javascript)
        app.router.add_post("/offer", offer)
        
        runner = web.AppRunner(app)
        await runner.setup()
        for port in range(8080, 8100):  # Try ports 8080 to 8099
            try:
                site = web.TCPSite(runner, "0.0.0.0", port)
                await site.start()
                local_ip = self.get_local_ip()
                print("WebRTC server started successfully.")
                print(f"Access it locally at: http://localhost:{port}")
                print(f"Access it from other devices on the same network at: http://{local_ip}:{port}")
                
                # Keep the server running
                while True:
                    await asyncio.sleep(1)
                
            except OSError as e:
                print(f"Port {port} is not available: {e}")
        
        print("Failed to start WebRTC server: No available ports")

    @staticmethod
    def get_local_ip():
        try:
            # This creates a UDP socket (doesn't actually connect to 8.8.8.8)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "127.0.0.1"  # Fallback to localhost if unable to determine IP
        
    async def on_cam_move(self, event, session, fps=60):
        # only intercept the ego camera.
        # if event.key != "ego":
        #     return
        try:
            # with self.head_matrix_shared.get_lock():  # Use the lock to ensure thread-safe updates
            #     self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            # with self.aspect_shared.get_lock():
            #     self.aspect_shared.value = event.value['camera']['aspect']
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass
        # self.head_matrix = np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F")
        # print(np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F"))
        # print("camera moved", event.value["matrix"].shape, event.value["matrix"])

    async def on_hand_move(self, event, session, fps=60):
        try:
            # with self.left_hand_shared.get_lock():  # Use the lock to ensure thread-safe updates
            #     self.left_hand_shared[:] = event.value["leftHand"]
            # with self.right_hand_shared.get_lock():
            #     self.right_hand_shared[:] = event.value["rightHand"]
            # with self.left_landmarks_shared.get_lock():
            #     self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
            # with self.right_landmarks_shared.get_lock():
            #     self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
            self.left_hand_shared[:] = event.value["leftHand"]
            self.right_hand_shared[:] = event.value["rightHand"]
            self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
            self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
        except:
            pass
    
    async def main_webrtc(self, session, fps=60):
        session.set @ DefaultScene(frameloop="always")
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        session.upsert @ WebRTCStereoVideoPlane(
                src="https://192.168.50.183:8080",
                # iceServer={},
                key="zed",
                aspect=1.33334,
                height = 8,
                position=[0, -2, -0.2],
            )
        while True:
            await asyncio.sleep(1)
    
    async def main_image(self, session, fps=60):
        session.upsert @ Hands(fps=fps, stream=True, key="hands")
        # session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        end_time = time.time()
        print('1111111111111111111111111111111111111111111')
        while True:
            if interrupt.interrupt_callback():
                logging.info("main_image detect interrupt")
                break

            start = time.time()
            # print(end_time - start)
            # aspect = self.aspect_shared.value
            display_image = self.img_array

            # session.upsert(
            # ImageBackground(
            #     # Can scale the images down.
            #     display_image[:self.img_height],
            #     # 'jpg' encoding is significantly faster than 'png'.
            #     format="jpeg",
            #     quality=80,
            #     key="left-image",
            #     interpolate=True,
            #     # fixed=True,
            #     aspect=1.778,
            #     distanceToCamera=2,
            #     position=[0, -0.5, -2],
            #     rotation=[0, 0, 0],
            # ),
            # to="bgChildren",
            # )

            session.upsert(
            [ImageBackground(
                # Can scale the images down.
                display_image[0, 0],
                # display_image[self.img_height::2, ::2],
                # 'jpg' encoding is significantly faster than 'png'.
                format="jpeg",
                quality=80,
                key="left-image",
                interpolate=True,
                # fixed=True,
                aspect=1.66667,
                # distanceToCamera=0.5,
                height=8,
                position=[0, -1, 3],
                # rotation=[0, 0, 0],
                layers=1,
            ),
                ImageBackground(
                    # Can scale the images down.
                    display_image[0, 1],
                    # display_image[self.img_height::2, ::2],
                    # 'jpg' encoding is significantly faster than 'png'.
                    format="jpeg",
                    quality=80,
                    key="right-image",
                    interpolate=True,
                    # fixed=True,
                    aspect=1.66667,
                    # distanceToCamera=0.5,
                    height=8,
                    position=[0, -1, 3],
                    # rotation=[0, 0, 0],
                    layers=2,
                )
            ],
            to="bgChildren",
            )
            # rest_time = 1/fps - time.time() + start
            end_time = time.time()
            await asyncio.sleep(0.03)

    def process(self, data):
        if data is not None and CONFIG_KEY_ISAAC_SIM in data:
            isaac_sim_data = data[CONFIG_KEY_ISAAC_SIM]
            if isinstance(isaac_sim_data, tuple) and len(isaac_sim_data) == 2:
                _, (_, image) = isaac_sim_data
                if image is not None and image.shape[0] > 0:
                    np.copyto(self.img_array, image)
        
    def run_webrtc_server(self):
        try:
            asyncio.run(self.start_webrtc_server())
        except Exception as e:
            logging.error(f"Error in WebRTC server: {e}")
            logging.error(traceback.format_exc())
                    
    def run(self):
        # Start WebRTC server in a separate thread
        webrtc_thread = threading.Thread(target=self.run_webrtc_server)
        webrtc_thread.start()

        # Run the main processing loop
        super().run()

