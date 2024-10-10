import cv2
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, VideoFrame  # Import VideoFrame
import aiohttp
import json

class VideoReceiver(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        return frame

async def display_video(track):
    video_receiver = VideoReceiver(track)
    while True:
        frame = await video_receiver.recv()
        frame_image = frame.to_ndarray()
        cv2.imshow("Received Frame", frame_image)

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

async def start():
    pc = RTCPeerConnection()

    # Set up WebRTC track to receive video stream
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            print("Receiving video track")
            asyncio.create_task(display_video(track))

    # Create an offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Send offer to the Raspberry Pi
    async with aiohttp.ClientSession() as session:
        async with session.post("http://192.168.2.165:8080/offer", json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }) as response:
            answer = await response.json()
            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

if __name__ == "__main__":
    asyncio.run(start())
    cv2.destroyAllWindows()
