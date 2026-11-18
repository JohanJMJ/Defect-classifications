"""
Camera Streamer for WebCam / USB Camera / Raspberry Pi Camera.
Provides on-demand camera access (camera stays OFF until the Live tab is opened),
continuous frame streaming, and graceful resource release.
"""

import threading
import time
import cv2
import numpy as np


class CameraStreamer:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.lock = threading.Lock()
        self.active_viewers = 0
        self.last_frame = None
        # Camera is deliberately NOT opened here to prevent the camera LED/hardware
        # from being powered on when the server starts. It is activated on-demand.
        print(f"[CameraStreamer] Initialized in STANDBY mode (Hardware camera is OFF).")

    def open_camera(self):
        """Activates physical camera hardware if not already open."""
        with self.lock:
            if self.cap is not None and self.cap.isOpened():
                return True

            try:
                # Open video capture device
                self.cap = cv2.VideoCapture(self.camera_index)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    print(f"[CameraStreamer] Physical camera index {self.camera_index} is now ACTIVE (Hardware ON).")
                    return True
                else:
                    self.cap = None
                    print(f"[CameraStreamer] Physical camera not available at index {self.camera_index}.")
                    return False
            except Exception as e:
                print(f"[CameraStreamer] Failed to open camera: {e}")
                self.cap = None
                return False

    def release(self):
        """Immediately turns off camera hardware and frees device handle."""
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                print("[CameraStreamer] Physical camera RELEASED (Hardware OFF).")
            self.active_viewers = 0

    def is_active(self):
        with self.lock:
            return self.cap is not None and self.cap.isOpened()

    def get_frame(self):
        with self.lock:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame = frame
                    return frame

        return self._create_test_card()

    def capture_snapshot(self):
        """
        Captures a single BGR frame for inspection.
        If camera is currently idle, opens it momentarily for snapshot and releases.
        """
        should_release = False
        with self.lock:
            camera_was_idle = (self.cap is None or not self.cap.isOpened())

        if camera_was_idle:
            success = self.open_camera()
            should_release = True
            # Allow camera sensor automatic exposure to settle
            time.sleep(0.15)

        frame = self.get_frame()
        result = frame.copy() if frame is not None else self._create_test_card()

        if should_release and self.active_viewers == 0:
            self.release()

        return result

    def _create_test_card(self):
        """Generates an industrial test pattern frame if camera is standby or unavailable."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Industrial dark gradient
        for y in range(self.height):
            c = int(18 + 24 * (y / self.height))
            img[y, :] = (c, c, c + 8)

        cx, cy = self.width // 2, self.height // 2
        # Crosshairs & target reticle
        cv2.line(img, (cx - 45, cy), (cx + 45, cy), (0, 200, 255), 1)
        cv2.line(img, (cx, cy - 45), (cx, cy + 45), (0, 200, 255), 1)
        cv2.circle(img, (cx, cy), 55, (0, 200, 255), 1)
        cv2.circle(img, (cx, cy), 6, (0, 180, 255), -1)

        t_str = time.strftime("%H:%M:%S")
        cv2.putText(img, "OPTICAL FEED: STANDBY", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, f"Status: Camera Idle | Switch to Live Tab or Click 'Start Camera'", (30, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        cv2.putText(img, f"System Clock: {t_str}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (6, 182, 212), 1)
        return img

    def generate_mjpeg(self):
        """Generator for Flask MJPEG streaming endpoint with ref-counting."""
        self.open_camera()
        with self.lock:
            self.active_viewers += 1

        try:
            while True:
                with self.lock:
                    if self.active_viewers <= 0:
                        break
                frame = self.get_frame()
                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.04)  # ~25 FPS
        finally:
            with self.lock:
                self.active_viewers = max(0, self.active_viewers - 1)
                if self.active_viewers == 0:
                    # No one is watching the stream anymore: release camera hardware!
                    if self.cap is not None:
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = None
                        print("[CameraStreamer] Stream ended and 0 active viewers. Camera RELEASED (Hardware OFF).")
