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
                # On Linux/Raspberry Pi, prefer V4L2 backend for lower latency and direct driver control
                import platform
                if platform.system() == "Linux":
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
                    if self.cap and self.cap.isOpened():
                        # Set MJPG codec on Linux/Pi to ensure full framerate and bandwidth headroom
                        try:
                            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        except Exception:
                            pass
                else:
                    self.cap = cv2.VideoCapture(self.camera_index)

                if not (self.cap and self.cap.isOpened()):
                    self.cap = cv2.VideoCapture(self.camera_index)

                if self.cap.isOpened():
                    # Set resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    # Minimize internal driver buffer to prevent stale / delayed frames
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
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
                if ret and frame is not None:
                    self.last_frame = frame
                    return frame
            # If camera frame drop occurred but a recent frame is cached, keep video feed continuous
            if self.last_frame is not None:
                return self.last_frame

        return self._create_test_card()

    def capture_snapshot(self):
        """
        Captures a single real BGR frame for AI inspection.
        - If live stream is active, instantly returns the latest live frame (zero contention/latency).
        - If camera is currently idle, opens it momentarily, flushes buffer for AE/AWB settling,
          captures the frame, and safely releases.
        - Returns None if physical camera is unavailable (NEVER returns the synthetic test card).
        """
        # Fast-Path: If live stream is actively producing frames, grab the newest frame directly!
        with self.lock:
            if self.cap is not None and self.cap.isOpened() and self.last_frame is not None:
                return self.last_frame.copy()

        # Slow-Path: Camera is idle or hasn't produced a frame yet
        should_release = False
        with self.lock:
            camera_was_idle = (self.cap is None or not self.cap.isOpened())

        if camera_was_idle:
            success = self.open_camera()
            if not success:
                return None
            should_release = True
            # Flush initial buffer frames so sensor auto-exposure (AE) and auto-white-balance (AWB) settle
            if self.cap is not None:
                for _ in range(8):
                    ret, f = self.cap.read()
                    if ret and f is not None:
                        self.last_frame = f
                    time.sleep(0.04)

        result = None
        with self.lock:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.last_frame = frame
                    result = frame.copy()
            if result is None and self.last_frame is not None:
                result = self.last_frame.copy()

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
