import cv2
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, VideoFrame  # Import VideoFrame
from aiohttp import web
import json
import numpy as np

class VideoCameraTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        # Capture frame from USB camera
        ret, frame = self.cap.read()

        # Resize or modify frame as needed
        if not ret:
            return

        # Convert frame to RGB (WebRTC expects RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a video frame object
        frame_data = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        frame_data.pts = pts
        frame_data.time_base = time_base
        return frame_data

# WebRTC signaling handler
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc.addTrack(VideoCameraTrack())  # Add video track

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print("ICE connection state is %s" % pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(content_type="application/json",
                        text=json.dumps({
                            "sdp": pc.localDescription.sdp,
                            "type": pc.localDescription.type
                        }))

# Run signaling server
async def run_server():
    app = web.Application()
    app.router.add_post("/offer", offer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "192.168.2.165", 8080)
    await site.start()
    print("Server started at http://192.168.2.165:8080")

if __name__ == "__main__":
    asyncio.run(run_server())
