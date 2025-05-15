import cv2
import mediapipe as mp
import os
import time
import numpy as np

# Directory to store iris data
DATA_DIR = "static/iris_data"
TEMP_DIR = "static/temp"

# Create directories if they don't exist
for directory in [DATA_DIR, TEMP_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Check write permissions for TEMP_DIR and DATA_DIR
for directory in [TEMP_DIR, DATA_DIR]:
    try:
        test_file = os.path.join(directory, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"Write permissions verified for {directory}")
    except Exception as e:
        print(f"Error: Cannot write to {directory}. Please check permissions: {e}")
        raise

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Iris landmarks for left and right eyes
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Function to get iris center
def get_iris_center(landmarks, iris_indices, shape):
    h, w = shape
    x_coords = [landmarks.landmark[i].x for i in iris_indices]
    y_coords = [landmarks.landmark[i].y for i in iris_indices]
    cx = min(max(int(sum(x_coords) / len(x_coords) * w), 0), w - 1)
    cy = min(max(int(sum(y_coords) / len(y_coords) * h), 0), h - 1)
    print(f"Calculated iris center - x: {cx}, y: {cy}")
    return cx, cy

# Function to crop the eye region
def crop_eye_region(frame, cx, cy, size=40):
    h, w, _ = frame.shape
    x1, y1 = max(cx - size, 0), max(cy - size, 0)
    x2, y2 = min(cx + size, w), min(cy + size, h)
    cropped = frame[y1:y2, x1:x2]
    print(f"Crop region - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, size: {cropped.size}")
    if cropped.size == 0:
        print("Warning: Cropped region is empty, returning default region")
        center_x, center_y = w // 2, h // 2
        default_x1, default_y1 = center_x - size, center_y - size
        default_x2, default_y2 = center_x + size, center_y + size
        cropped = frame[default_y1:default_y2, default_x1:default_x2]
        if cropped.size == 0:
            cropped = np.zeros((size * 2, size * 2, 3), dtype=np.uint8)
    return cropped

# Function to compare two images using Mean Squared Error (MSE)
def compare_images(image1, image2):
    image1_resized = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    diff = cv2.absdiff(image1_resized, image2)
    mse = np.sum(diff ** 2) / float(image1_resized.shape[0] * image1_resized.shape[1])
    return mse

# Function to draw a progress bar
def draw_progress_bar(frame, progress, x=30, y=60, w=200, h=20):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), -1)
    progress_width = int(w * (progress / 100))
    cv2.rectangle(frame, (x, y), (x + progress_width, y + h), (0, 255, 0), -1)
    cv2.putText(frame, f"Progress: {progress}%", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Function to find an available camera
def find_available_camera(max_index=10, max_attempts=5, preferred_index=None):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    if preferred_index is not None:
        for backend in backends:
            backend_name = "CAP_DSHOW" if backend == cv2.CAP_DSHOW else "CAP_ANY"
            print(f"Trying preferred camera index {preferred_index} with backend {backend_name}")
            cap = cv2.VideoCapture(preferred_index, backend)
            if cap.isOpened():
                print(f"Camera found at preferred index {preferred_index} with backend {backend_name}")
                return cap, preferred_index
            cap.release()
            time.sleep(2.0)

    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1} to find camera...")
        for backend in backends:
            backend_name = "CAP_DSHOW" if backend == cv2.CAP_DSHOW else "CAP_ANY"
            print(f"Trying backend: {backend_name}")
            for index in range(max_index):
                print(f"Trying camera index {index} with backend {backend_name}")
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    print(f"Camera found at index {index} with backend {backend_name} on attempt {attempt + 1}")
                    return cap, index
                else:
                    print(f"Failed to open camera at index {index} with backend {backend_name}")
                cap.release()
                time.sleep(2.0)
            print(f"Trying camera index -1 with backend {backend_name}")
            cap = cv2.VideoCapture(-1, backend)
            if cap.isOpened():
                print(f"Camera found at index -1 with backend {backend_name} on attempt {attempt + 1}")
                return cap, -1
            else:
                print(f"Failed to open camera at index -1 with backend {backend_name}")
            cap.release()
            time.sleep(2.0)
    print("Error: No available camera found after multiple attempts. Troubleshooting tips:")
    print("- Ensure the camera is physically connected and powered on.")
    print("- Check Device Manager to confirm the camera is recognized.")
    print("- Verify no other applications are using the camera (e.g., Zoom, Skype).")
    print("- Replug the camera or try a different USB port.")
    print("- Restart your system to reset the camera driver.")
    print("- Check camera permissions in Settings > Privacy > Camera.")
    print("- Test the camera with another application (e.g., Windows Camera app).")
    return None, None

# Function to capture and save iris images during registration
def capture_and_save_iris(username):
    cv2.destroyAllWindows()

    # Use the camera index that worked in the test script (e.g., 0, 1, etc.)
    cap, camera_index = find_available_camera(preferred_index=0)
    if cap is None:
        print("Error: No available camera found. Please check camera connection, permissions, or close other applications using the camera.")
        cv2.destroyAllWindows()
        raise Exception("Failed to initialize camera. Please check camera connection, permissions, or close other applications using the camera.")

    print(f"Using camera at index {camera_index}")
    saved = False
    start_time = time.time()
    max_duration = 30
    cancel_signal = os.path.join(TEMP_DIR, f"{username}_cancel")
    window_name = "Iris Capture"

    print("Creating camera window...")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.moveWindow(window_name, 100, 100)
    print(f"Created window: {window_name}")

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "Testing window...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, blank_frame)
    cv2.waitKey(1000)
    print("Tested window display")

    try:
        h, w = 480, 640
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            print(f"Frame {frame_count} captured successfully")
            frame_count += 1

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            indicator_color = (0, 0, 255)
            label_text = "Not Detected"
            lcx, lcy, rcx, rcy = w // 4, h // 2, 3 * w // 4, h // 2

            face_detected = False
            if results.multi_face_landmarks:
                print("Face detected")
                face_detected = True
                for face_landmarks in results.multi_face_landmarks:
                    for i in LEFT_IRIS + RIGHT_IRIS:
                        print(f"Landmark {i}: x={face_landmarks.landmark[i].x}, y={face_landmarks.landmark[i].y}")

                    # Calculate iris centers (similar to test script)
                    lcx = int(sum([face_landmarks.landmark[i].x for i in LEFT_IRIS]) / 4 * w)
                    lcy = int(sum([face_landmarks.landmark[i].y for i in LEFT_IRIS]) / 4 * h)
                    rcx = int(sum([face_landmarks.landmark[i].x for i in RIGHT_IRIS]) / 4 * w)
                    rcy = int(sum([face_landmarks.landmark[i].y for i in RIGHT_IRIS]) / 4 * h)
                    print(f"Iris coordinates - Left: ({lcx}, {lcy}), Right: ({rcx}, {rcy})")

                    indicator_color = (0, 255, 0)
                    label_text = "Detected"

            else:
                print("No face detected, using default positions for indicators")

            # Draw eyeball indicators (green circles)
            cv2.circle(frame, (lcx, lcy), 5, (0, 255, 0), 2)
            cv2.circle(frame, (rcx, rcy), 5, (0, 255, 0), 2)

            cv2.putText(frame, label_text, (lcx - 70, lcy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, indicator_color, 2)
            cv2.putText(frame, label_text, (rcx - 70, rcy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, indicator_color, 2)

            elapsed_time = time.time() - start_time
            progress = min(int((elapsed_time / max_duration) * 100), 100)
            draw_progress_bar(frame, progress)

            # Attempt to capture iris images
            if face_detected and not saved:
                print("Attempting to capture iris images...")
                left_eye = crop_eye_region(frame, lcx, lcy)
                right_eye = crop_eye_region(frame, rcx, rcy)
                try:
                    print("Saving left eye image to temp...")
                    cv2.imwrite(f"{TEMP_DIR}/{username}_left.png", left_eye)
                    print("Saving right eye image to temp...")
                    cv2.imwrite(f"{TEMP_DIR}/{username}_right.png", right_eye)
                    print("Moving images to iris_data...")
                    cv2.imwrite(f"{DATA_DIR}/{username}_left.png", left_eye)
                    cv2.imwrite(f"{DATA_DIR}/{username}_right.png", right_eye)
                    saved = True
                    print(f"Iris images saved to {TEMP_DIR} and moved to {DATA_DIR}")
                    cv2.putText(frame, "Iris Captured!", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error saving iris images: {e}")
                    cv2.imwrite(f"{TEMP_DIR}/{username}_debug_frame.png", frame)
                    print(f"Saved debug frame to {TEMP_DIR}/{username}_debug_frame.png")

            if saved:
                cv2.putText(frame, "Capturing iris images... This may take a moment.t", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif face_detected:
                cv2.putText(frame, "Align face for iris scan", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected, align face", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(frame, "Press 'q' or click Cancel to stop", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            try:
                cv2.imshow(window_name, frame)
                print("Displayed frame in window")
            except Exception as e:
                print(f"Warning: Failed to display frame in window: {e}")
                print("Capturing will continue without displaying the window.")

            if os.path.exists(cancel_signal):
                print("Capture cancelled by user.")
                break

            if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time > max_duration:
                break

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            time.sleep(3.0)
            print("Camera release completed with 3s delay")
        cv2.destroyWindow(window_name)
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

    return saved and not os.path.exists(cancel_signal)

# Function to compare iris for login
def compare_iris(username):
    cv2.destroyAllWindows()

    print("Attempting to release any existing camera resources...")
    temp_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if temp_cap.isOpened():
        temp_cap.release()
        time.sleep(2.0)
        print("Released temporary camera instance")

    cap, camera_index = find_available_camera(preferred_index=0)
    if cap is None:
        print("Error: No available camera found. Please check camera connection, permissions, close other applications using the camera, replug the camera, or restart your system.")
        cv2.destroyAllWindows()
        raise Exception("Failed to initialize camera. Please check camera connection, permissions, or close other applications using the camera.")

    print(f"Using camera at index {camera_index}")
    start_time = time.time()
    max_duration = 30
    captured = False

    user_left_path = os.path.join(DATA_DIR, f"{username}_left.png")
    user_right_path = os.path.join(DATA_DIR, f"{username}_right.png")

    if not (os.path.exists(user_left_path) and os.path.exists(user_right_path)):
        print("Error: User iris data not found.")
        cap.release()
        cv2.destroyAllWindows()
        return False

    left_ref = cv2.imread(user_left_path)
    right_ref = cv2.imread(user_right_path)

    window_name = "Login Iris Scanner"
    print("Creating camera window...")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.moveWindow(window_name, 100, 100)
    print(f"Created window: {window_name}")

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "Testing window...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, blank_frame)
    cv2.waitKey(1000)
    print("Tested window display")

    try:
        h, w = 480, 640
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            indicator_color = (0, 0, 255)
            label_text = "Not Detected"
            lcx, lcy, rcx, rcy = w // 4, h // 2, 3 * w // 4, h // 2

            face_detected = False
            if results.multi_face_landmarks:
                print("Face detected")
                face_detected = True
                for face_landmarks in results.multi_face_landmarks:
                    for i in LEFT_IRIS + RIGHT_IRIS:
                        print(f"Landmark {i}: x={face_landmarks.landmark[i].x}, y={face_landmarks.landmark[i].y}")

                    lcx = int(sum([face_landmarks.landmark[i].x for i in LEFT_IRIS]) / 4 * w)
                    lcy = int(sum([face_landmarks.landmark[i].y for i in LEFT_IRIS]) / 4 * h)
                    rcx = int(sum([face_landmarks.landmark[i].x for i in RIGHT_IRIS]) / 4 * w)
                    rcy = int(sum([face_landmarks.landmark[i].y for i in RIGHT_IRIS]) / 4 * h)
                    print(f"Iris coordinates - Left: ({lcx}, {lcy}), Right: ({rcx}, {rcy})")

                    indicator_color = (0, 255, 0)
                    label_text = "Detected"

            else:
                print("No face detected, using default positions for indicators")

            cv2.circle(frame, (lcx, lcy), 5, (0, 255, 0), 2)
            cv2.circle(frame, (rcx, rcy), 5, (0, 255, 0), 2)

            cv2.putText(frame, label_text, (lcx - 70, lcy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, indicator_color, 2)
            cv2.putText(frame, label_text, (rcx - 70, rcy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, indicator_color, 2)

            elapsed = time.time() - start_time
            progress = min(int((elapsed / max_duration) * 100), 100)
            draw_progress_bar(frame, progress)

            if face_detected and not captured:
                left_eye = crop_eye_region(frame, lcx, lcy)
                right_eye = crop_eye_region(frame, rcx, rcy)

                score_left = compare_images(left_ref, left_eye)
                score_right = compare_images(right_ref, right_eye)

                print(f"Similarity scores: Left={score_left:.2f}, Right={score_right:.2f}")
                if score_left < 500 and score_right < 500:
                    captured = True
                    cv2.putText(frame, "Login Successful!", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Align face for iris scan", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected, align face", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            try:
                cv2.imshow(window_name, frame)
                print("Displayed frame in window")
            except Exception as e:
                print(f"Warning: Failed to display frame in window: {e}")
                print("Capturing will continue without displaying the window.")

            if cv2.waitKey(1) & 0xFF == ord('q') or elapsed > max_duration or captured:
                break

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            time.sleep(3.0)
            print("Camera release completed with 3s delay")
        cv2.destroyWindow(window_name)
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

    return captured