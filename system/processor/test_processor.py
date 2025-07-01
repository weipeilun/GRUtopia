# encoding: utf-8

from logging import INFO
import os
import cv2
from system.processor.abstract_processor import AbstractProcessor
from system.constants import CONFIG_KEY_TEST, CONFIG_KEY_ISAAC_SIM, CONFIG_KEY_IN_QUEUE
from system.tools.interrupt import interrupt_callback
from system.tools.processor_config import config
import time
import asyncio
import traceback
import threading
import json
import queue
import socket
import logging
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaPlayer
from av import VideoFrame
import numpy as np
from aiohttp import web
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
# from aiortc.contrib.signaling import BridgeOfferAnswerSignaling
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender
from system.tools import interrupt
from multiprocessing import Event


class SteroVideoTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, img_queue, toggle_streaming, fps):
        super().__init__()  # Initialize base class
        # self.img_shape = (2*img_shape[0], img_shape[1], 3)
        # self.img_height, self.img_width = img_shape[:2]
        # self.shm_name = shm_name
        # existing_shm = shared_memory.SharedMemory(name=shm_name)
        # self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
        self.img_queue = img_queue
        self.toggle_streaming = toggle_streaming
        self.streaming_started = False
        self.timescale = 1000  # Use a timescale of 1000 for milliseconds
        # self.frame_interval = 1 / fps
        self._last_frame_time = time.time()
        self.start_time = time.time()
    
    async def recv(self):
        """
        This method is called when a new frame is needed.
        """
        # now = time.time()
        # wait_time = self._last_frame_time + self.frame_interval - now
        # if wait_time > 0:
        #     await asyncio.sleep(wait_time)
        # self._last_frame_time = time.time()
        # start = time.time()
        if not self.streaming_started:
            self.toggle_streaming.set()
            self.streaming_started = True
        frame = self.img_queue.get()
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


class TestProcessor(AbstractProcessor):
    _PID = CONFIG_KEY_TEST
        
    def __init__(self, in_queue_list=None, out_queue_info=None, interrupt_check=interrupt.interrupt_callback, listen_interval=0.01, do_validate=True, debug_time=False, log_level=logging.INFO):
        super().__init__(in_queue_list=in_queue_list, out_queue_info=out_queue_info, interrupt_check=interrupt_check, listen_interval=listen_interval, do_validate=do_validate, debug_time=debug_time, log_level=log_level)
        self.pcs = set()
        
        self.img_queue = queue.Queue()
        self.toggle_streaming = True
        self.fps = 30
        self.toggle_streaming = Event()
        # self.signaling = BridgeOfferAnswerSignaling()
        
    async def start_webrtc_server(self):

        # async def offer(request):
        #     print("Received offer request")
        #     params = await request.json()
        #     offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        #     pc = RTCPeerConnection()
        #     self.pcs.add(pc)

        #     @pc.on("connectionstatechange")
        #     async def on_connectionstatechange():
        #         print(f"Connection state is {pc.connectionState}")
        #         if pc.connectionState == "failed":
        #             await pc.close()
        #             self.pcs.discard(pc)

        #     pc.addTrack(self.left_track)
        #     pc.addTrack(self.right_track)

        #     await pc.setRemoteDescription(offer)
        #     answer = await pc.createAnswer()
        #     await pc.setLocalDescription(answer)

        #     return web.Response(
        #         content_type="application/json",
        #         text=json.dumps({
        #             "sdp": pc.localDescription.sdp,
        #             "type": pc.localDescription.type
        #         })
        #     )
        

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
            zed_track = SteroVideoTrack(self.img_queue, self.toggle_streaming, self.fps)
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
                print(f"WebRTC server started successfully.")
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

    def process(self, data):
        # print(time.time())
        for target_pid in config[CONFIG_KEY_TEST][CONFIG_KEY_IN_QUEUE]:
            if data is not None and target_pid in data:
                isaac_sim_data = data[target_pid]
            if isinstance(isaac_sim_data, tuple) and len(isaac_sim_data) == 2:
                _, (_, image) = isaac_sim_data
                if image is not None and image.shape[0] > 0:
                    self.img_queue.put(image)
                    # # Convert BGR to RGB
                    # left_image_rgb = cv2.cvtColor(image[0, 0], cv2.COLOR_BGR2RGB)
                    # right_image_rgb = cv2.cvtColor(image[0, 1], cv2.COLOR_BGR2RGB)
                    
                    # # cv2.imshow('left', left_image_rgb)
                    # # cv2.imshow('right', right_image_rgb)
                    # # cv2.waitKey(1)
                    # self.left_track.frame = left_image_rgb
                    # self.right_track.frame = right_image_rgb

                    # print('11111111111111111111111111111111111111')
        
    def run_webrtc_server(self):
        try:
            asyncio.run(self.start_webrtc_server())
        except Exception as e:
            logging.error(f"Error in WebRTC server: {e}")
            logging.error(traceback.format_exc())

    def run(self):
        logging.basicConfig(level=logging.DEBUG)
        
        # Start WebRTC server in a separate thread
        webrtc_thread = threading.Thread(target=self.run_webrtc_server)
        webrtc_thread.start()

        # Run the main processing loop
        super().run()
