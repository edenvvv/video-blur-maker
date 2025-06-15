import cv2
import numpy as np
import time
import base64
import argparse
import sys
from datetime import datetime
from multiprocessing import Process, Queue, set_start_method
import signal
import os


def init_background_subtractor():
    """Initialize background subtractor inside the process"""
    return cv2.createBackgroundSubtractorMOG2(detectShadows=True)


def streamer_process(video_path, output_queue):
    """Streamer process function"""
    print(f"Streamer from: Starting video stream from {video_path}")  # in the future write to log

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")  # in the future write to log
        output_queue.put({'type': 'error', 'message': 'Could not open video'})
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Streamer: End of video reached")  # in the future write to log
                output_queue.put({'type': 'end'})
                break

            # encode frame to base64 for transmission
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            message = {
                'type': 'frame',
                'frame': frame_base64,
                'frame_number': frame_count,
                'timestamp': time.time()
            }

            # Send to detector
            try:
                output_queue.put(message, timeout=0.1)
            except Exception as e:
                print(f"Streamer: Queue full, skipping frame, more detail at ({e})")  # in the future write to log

            frame_count += 1

            time.sleep(frame_delay)

    except Exception as e:
        print(f"Streamer error: {e}")  # in the future write to log
    finally:
        cap.release()
        print("Streamer: Finished")  # in the future write to log


# ==================== DETECTOR COMPONENT ====================
def detector_process(input_queue, output_queue):
    """Detector process function"""
    print("Detector: Starting motion detection")  # in the future write to log

    # Initialize background subtractor inside this process
    background_subtractor = init_background_subtractor()

    def decode_frame(frame_base64):
        """Decode base64 frame"""
        frame_data = base64.b64decode(frame_base64)
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        return frame

    def detect_motion(frame):
        """Detect motion in frame using background subtraction - return all detections above threshold"""
        # Apply background subtraction
        fg_mask = background_subtractor.apply(frame)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        min_area = 500

        # Return all detections above minimum area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': int(area)
                })

        return detections

    try:
        while True:
            try:
                message = input_queue.get(timeout=1.0)
            except:
                continue

            if message['type'] == 'end':
                print("Detector: Received end signal")  # in the future write to log
                output_queue.put({'type': 'end'})
                break
            elif message['type'] == 'error':
                output_queue.put(message)
                break

            if message['type'] == 'frame':
                # Decode frame
                frame = decode_frame(message['frame'])

                # Detect motion
                detections = detect_motion(frame)

                # Send to displayer
                response = {
                    'type': 'frame_with_detections',
                    'frame': message['frame'],  # Forward original frame
                    'detections': detections,
                    'frame_number': message['frame_number'],
                    'timestamp': message['timestamp']
                }

                try:
                    output_queue.put(response, timeout=0.1)
                except:
                    print("Detector: Display queue full")  # in the future write to log

    except Exception as e:
        print(f"Detector error: {e}")  # in the future write to log
    finally:
        print("Detector: Finished")  # in the future write to log


# ==================== DISPLAYER COMPONENT ====================
def displayer_process(input_queue):
    """Displayer process function"""
    print("Displayer: Starting display")  # in the future write to log

    def decode_frame(frame_base64):
        """Decode base64 frame"""
        frame_data = base64.b64decode(frame_base64)
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        return frame

    def blur_detections(frame, detections):
        """Apply blur to detected motion areas"""
        for detection in detections:
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']

            # Ensure coordinates are within frame bounds
            frame_height, frame_width = frame.shape[:2]
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            x2 = max(0, min(x + w, frame_width))
            y2 = max(0, min(y + h, frame_height))

            # Extract the region of interest
            if x2 > x and y2 > y:
                roi = frame[y:y2, x:x2]

                # Apply Gaussian blur - adjust kernel size for blur intensity
                blur_kernel_size = 31
                blurred_roi = cv2.GaussianBlur(roi, (blur_kernel_size, blur_kernel_size), 0)

                # Replace the original region with blurred version
                frame[y:y2, x:x2] = blurred_roi

        return frame

    def draw_detections(frame, detections):
        """Draw bounding boxes around detected motion"""
        for detection in detections:
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw area text
            cv2.putText(
                frame,
                f"Area: {detection['area']}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        return frame

    def draw_timestamp(frame):
        """Draw current time in upper left corner"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame,
            timestamp,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        return frame

    try:
        while True:
            try:
                message = input_queue.get(timeout=1.0)
            except:
                continue

            if message['type'] == 'end':
                print("Displayer: Video ended - shutting down display")  # in the future write to log
                break
            elif message['type'] == 'error':
                print(f"Displayer: Received error - {message['message']}")  # in the future write to log
                break

            if message['type'] == 'frame_with_detections':
                # Decode frame
                frame = decode_frame(message['frame'])

                # apply blur to detected areas
                frame = blur_detections(frame, message['detections'])

                # then draw detection boxes
                frame = draw_detections(frame, message['detections'])

                # draw timestamp
                frame = draw_timestamp(frame)

                cv2.imshow('Video Analytics', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Displayer: User requested quit")  # in the future write to log
                    break

    except Exception as e:
        print(f"Displayer error: {e}")  # in the future write to log
    finally:
        cv2.destroyAllWindows()
        print("Displayer: Shutdown complete")  # in the future write to log


# ==================== MAIN SYSTEM ====================
class VideoAnalyticsSystem:
    def __init__(self, video_path):
        self.video_path = video_path
        self.processes = []
        self.shutdown_requested = False

        # Create queues
        self.streamer_to_detector_queue = Queue(maxsize=10)
        self.detector_to_displayer_queue = Queue(maxsize=10)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("Shutdown signal received...")  # in the future write to log
        self.shutdown_requested = True
        self.stop()

    def start(self):
        """Start all components"""
        print("Starting Video Analytics System...")  # in the future write to log

        # setup signal handlers for shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # start processes
        streamer_proc = Process(
            target=streamer_process,
            args=(self.video_path, self.streamer_to_detector_queue)
        )
        detector_proc = Process(
            target=detector_process,
            args=(self.streamer_to_detector_queue, self.detector_to_displayer_queue)
        )
        displayer_proc = Process(
            target=displayer_process,
            args=(self.detector_to_displayer_queue,)
        )

        self.processes = [streamer_proc, detector_proc, displayer_proc]

        # start all
        for process in self.processes:
            process.start()

        try:
            while not self.shutdown_requested:
                # check if any process has finished
                all_alive = all(process.is_alive() for process in self.processes)

                if not all_alive:
                    print("Video processing completed - initiating system shutdown")  # in the future write to log
                    break

                time.sleep(0.1)

            # wait for the processes to complete
            for process in self.processes:
                process.join(timeout=5.0)

        except Exception as e:
            print(f"Error during processing: {e}")  # in the future write to log
        finally:
            self.stop()
            print("Video Analytics System: Complete shutdown finished")  # in the future write to log

    def stop(self):
        """Stop all processes forcefully if needed"""
        print("Stopping all processes...")
        for process in self.processes:
            if process.is_alive():
                print(f"Terminating process {process.name}")  # in the future write to log
                process.terminate()
                process.join(timeout=2.0)

                # Force kill if still alive
                if process.is_alive():
                    print(f"Force killing process {process.name}")  # in the future write to log
                    process.kill()
                    process.join()


def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='Video Analytics System')
    parser.add_argument('video_path', help='Path to input video file')
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print("use the default video (demo.mp4)")  # in the future write to log
        vid_path = "demo.mp4"
    else:
        vid_path = args.video_path

    system = VideoAnalyticsSystem(vid_path)

    try:
        system.start()
    except KeyboardInterrupt:
        print("Shutting down system...")  # in the future write to log
        system.stop()


if __name__ == "__main__":
    main()
