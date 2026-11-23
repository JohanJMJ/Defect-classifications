import threading
import time

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False


# GPIO Pins (BCM numbering) - Extra feature (LED's and Buzzer is not connected)
PIN_GREEN_LED = 17   # PASS
PIN_YELLOW_LED = 27  # MONITOR
PIN_RED_LED = 22     # REJECT
PIN_BUZZER = 23      # Audio alert


class AlertManager:
    def __init__(self, enable_gpio=True):
        self.enable_gpio = enable_gpio and GPIO_AVAILABLE
        self.last_alert = {
            "verdict": "PASS",
            "severity": "NONE",
            "timestamp": time.time(),
            "buzzer_active": False,
        }

        if self.enable_gpio:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                for pin in [PIN_GREEN_LED, PIN_YELLOW_LED, PIN_RED_LED, PIN_BUZZER]:
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)
                print("[AlertManager] GPIO Pins successfully initialized.")
            except Exception as e:
                print(f"[AlertManager] GPIO initialization failed: {e}")
                self.enable_gpio = False
        else:
            print("[AlertManager] Running with Simulated Alerts (No physical GPIO).")

    def trigger_inspection_alert(self, verdict, severity="LOW"):
        self.last_alert = {
            "verdict": verdict,
            "severity": severity,
            "timestamp": time.time(),
            "buzzer_active": (verdict == "REJECT"),
        }

        if not self.enable_gpio:
            return

        def _pulse():
            try:
                # Reset all
                for pin in [PIN_GREEN_LED, PIN_YELLOW_LED, PIN_RED_LED, PIN_BUZZER]:
                    GPIO.output(pin, GPIO.LOW)

                if verdict == "PASS":
                    GPIO.output(PIN_GREEN_LED, GPIO.HIGH)
                    time.sleep(1.0)
                    GPIO.output(PIN_GREEN_LED, GPIO.LOW)

                elif verdict in ["DIVERT", "INSPECT", "FLAGGED_MONITOR"]:
                    GPIO.output(PIN_YELLOW_LED, GPIO.HIGH)
                    time.sleep(1.5)
                    GPIO.output(PIN_YELLOW_LED, GPIO.LOW)

                elif verdict == "REJECT":
                    GPIO.output(PIN_RED_LED, GPIO.HIGH)
                    # 2 rapid buzzer beeps
                    for _ in range(2):
                        GPIO.output(PIN_BUZZER, GPIO.HIGH)
                        time.sleep(0.15)
                        GPIO.output(PIN_BUZZER, GPIO.LOW)
                        time.sleep(0.1)
                    time.sleep(1.5)
                    GPIO.output(PIN_RED_LED, GPIO.LOW)

            except Exception as e:
                print(f"[AlertManager] Error during GPIO pulse: {e}")

        # Run in separate thread so inspection flow never blocks
        threading.Thread(target=_pulse, daemon=True).start()

    def get_status(self):
        return {
            "gpio_available": self.enable_gpio,
            "last_alert": self.last_alert,
        }

    def cleanup(self):
        if self.enable_gpio:
            try:
                GPIO.cleanup()
            except Exception:
                pass
