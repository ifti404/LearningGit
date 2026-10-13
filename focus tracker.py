import cv2 # type: ignore
import time
import threading

class FocusTracker:
    def __init__(self, absence_threshold: int = 45, on_absent_callback=None):
        self.absence_threshold = absence_threshold
        self.callback = on_absent_callback
        self.running = False
        self.cap = None
        self._thread = None
        
        # Load OpenCV Haar Cascade for frontal face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier XML file.")

    def start(self):
        """Activates the webcam and starts the background tracking loop."""
        if self.running:
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam. Check hardware permissions.")
            
        self.running = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()
        print("[FocusTracker] Webcam activated. Tracking started.")

    def _tracking_loop(self):
        last_face_time = time.time()
        alert_triggered = False

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[FocusTracker] Warning: Failed to grab frame from webcam.")
                time.sleep(1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )

            current_time = time.time()

            if len(faces) > 0:
                # Face detected; reset timer
                last_face_time = current_time
                alert_triggered = False
            else:
                # Face absent
                elapsed_absence = current_time - last_face_time
                if elapsed_absence >= self.absence_threshold and not alert_triggered:
                    alert_triggered = True
                    print(f"[FocusTracker] Alert: Face absent for {elapsed_absence:.1f} seconds!")
                    if self.callback:
                        self.callback()

            # Small sleep to reduce CPU overhead
            time.sleep(0.1)

    def stop(self):
        """Stops tracking and safely releases the webcam resource."""
        if not self.running:
            return
        
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        cv2.destroyAllWindows()
        print("[FocusTracker] Webcam successfully released and tracker stopped.")

if __name__ == "__main__":
    def test_alert():
        print("--> CALLBACK TRIGGERED: User has been away from camera too long!")

    print("Initializing Focus Tracker Test (Absence Threshold = 10s for testing)...")
    tracker = FocusTracker(absence_threshold=10, on_absent_callback=test_alert)
    
    try:
        tracker.start()
        print("Tracker running. Step away from the camera for 10 seconds to test alert. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping tracker...")
        tracker.stop()