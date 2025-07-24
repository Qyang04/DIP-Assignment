import cv2
import numpy as np
from matplotlib import pyplot as plt

# Calculate the brightness of the video
def calculate_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale
    return np.mean(gray)

# Detect the video is taken during daytime or nightime
def classify_day_night(brightness_values, night_threshold=100):
    return np.mean(brightness_values) < night_threshold

# Brighten the video function
def brighten(frame, factor=1.6):
    return np.clip(frame * factor, 0, 255).astype(np.uint8)

def blur_faces(frame):
    face_cascade = cv2.CascadeClassifier("face_detector.xml")
    if face_cascade.empty():
        raise FileNotFoundError("face_detector.xml not found or invalid")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        x, y = max(0, x - w//5), max(0, y - h//5)
        w, h = min(frame.shape[1] - x, w + w//2), min(frame.shape[0] - y, h + h//2)

        # based on face size to add blur
        blur_size = max(w, h) // 2
        kernel_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        face_roi = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 30)

    return frame

# Function to resize the foreground video and overlay it on top of the background video
def resizeAndOverlayVideo(background, foreground, scale_percent):
    # Get the current frame size
    height, width, _ = foreground.shape

    # Resize the frame
    # Calculate the new dimensions based on scaling percentage
    new_width = int(width*scale_percent/100)
    new_height = int(height*scale_percent/100)
    new_frame_size = (new_width, new_height)

    resizedForegroundVideo = cv2.resize(foreground, new_frame_size, interpolation = cv2.INTER_AREA)

    # Add border around the resized foreground
    border_thickness = 5
    resizedForegroundVideo = cv2.copyMakeBorder(resizedForegroundVideo, top=border_thickness, bottom=border_thickness, left=border_thickness,
                                                right=border_thickness, borderType=cv2.BORDER_CONSTANT, value=0) #value = 0 means black (applies to all channels)

    # Adjust overlay dimensions to include border
    new_height += 2 * border_thickness
    new_width += 2 * border_thickness

    # Resize background if necessary to match overlay
    bg_height, bg_width = background.shape[:2]

    # Check if the background is smaller than the resized foreground
    # If so, resize the background to fit the overlay
    if bg_height < new_height or bg_width < new_width:
        background = cv2.resize(background, (max(new_width, bg_width), max(new_height, bg_height)))

    # Overlay the resized foreground onto the background at the top-left corner
    background[0:new_height, 0:new_width] = resizedForegroundVideo
    return background

# === Function to remove black background ===
def remove_black_background(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load BGR image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nrow, ncol = gray.shape

    result = np.zeros((nrow, ncol, 4), dtype=np.uint8)
    result[:, :, :3] = image  # Copy BGR

    for y in range(nrow):
        for x in range(ncol):
            result[y, x, 3] = 255 if gray[y, x] > 10 else 0  # Transparent background
    return result

# === Function to overlay transparent watermark ===
def overlay_transparent(frame, watermark):
    watermark = cv2.resize(watermark, (frame.shape[1], frame.shape[0]))
    alpha = watermark[:, :, 3] / 255.0
    for c in range(3):
        frame[:, :, c] = (1 - alpha) * frame[:, :, c] + alpha * watermark[:, :, c]
    return frame

# === Function to append endscreen ===
def add_endscreen(writer, endscreen_path, width, height):
    end_vid = cv2.VideoCapture(endscreen_path)
    while True:
        success_end, end_frame = end_vid.read()
        if not success_end:
            break
        end_frame = cv2.resize(end_frame, (width, height))
        writer.write(end_frame)
    end_vid.release()

def process_video(input_path, output_path, talking_path, watermark1_path, watermark2_path, end_screen_path):

    print(input_path)

    # Load videos and watermarks
    vid = cv2.VideoCapture(input_path)
    talking_vid = cv2.VideoCapture(talking_path)
    watermark1 = remove_black_background(watermark1_path)
    watermark2 = remove_black_background(watermark2_path)

    # Get video properties
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_no_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    total_no_foreground_frames = talking_vid.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = 0

    if total_no_frames < total_no_foreground_frames:
        total_frames = total_no_foreground_frames
    else:
        total_frames = total_no_frames

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Step 1: Day/Night Detection & Brightness Adjustment
    # Calculate brightness and classification daytime or nighttime
    brightness_values = [] # List to store brightness values for each frame
    for frame_count in range(0, int(total_no_frames)):
        success, frame = vid.read()
        if not success:
            break
        brightness_values.append(calculate_brightness(frame))

    # Detect if the video is taken during daytime or nighttime after analyzing all frames
    average_brightness = np.mean(brightness_values)
    is_night = classify_day_night(brightness_values) # If the average brightness is less than threshold value, then it is nighttime
    print(f"Average brightness of the video: {average_brightness:.2f}")

    # Display a message indicating the video is taken during nighttime or daytime and whether brightness adjustment is needed
    if is_night:
        print(f"The {input_path} video is taken during nighttime. Brightness value will be adjusted.")
    else:
        print(f"The {input_path} video is taken during daytime. No brightness value will be adjusted.")

    # Process each frame
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video
    frame_count = 0

    success, frame = vid.read()
    foregroundSuccess, foreground = talking_vid.read()

    scale_percent = 30
    lastBackgroundFrame = None

    while success or foregroundSuccess:
        # Handle background:
        # If the background has a new frame, update the last valid frame
        if success:
            lastBackgroundFrame = frame.copy()
        # If background ends, reuse the last valid background frame
        elif lastBackgroundFrame is not None:
            frame = lastBackgroundFrame
        else:
            print(f"Stopped at frame {frame_count} (may be incomplete video)")
            break # Exit if there's no background at all

        # If it is nightime, brighten the frame
        if is_night and frame is not None: 
            frame = brighten(frame)

        # Step 2: Face Blurring
        if frame is not None:
            frame = blur_faces(frame)

        # Step 3: Overlay Talking Video
        # Call the function with input and output file paths
        # Scale percentage to shrink the foreground video
        # Process frames until both videos have ended

        # If there is a foreground frame, resize and overlay it
        if foregroundSuccess:
            frame = resizeAndOverlayVideo(frame, foreground, scale_percent)

        if frame is not None:
            if ((frame_count // int (fps * 4)) % 2) == 0:
                frame = overlay_transparent(frame, watermark1)
            else:
                frame = overlay_transparent(frame, watermark2)

        # Write processed frame
        if frame is not None:
            out.write(frame)

        success, frame = vid.read()
        foregroundSuccess, foreground = talking_vid.read()
        frame_count += 1

        if frame_count % 100 == 0 or frame_count == int(total_frames)-1:
            if frame_count < len(brightness_values):
                brightness_info = f"Brightness: {brightness_values[frame_count]:.1f}"
            else:
                brightness_info = "Brightness: N/A"
            print(f"Processing frame {frame_count}/{int(total_frames)} | {brightness_info}")

    # Step 5: Append End Screen
    add_endscreen(out, end_screen_path, width, height)

    # Release resources
    vid.release()
    talking_vid.release()
    out.release()

    # Plot the histogram to visualize the brightness of each video
    plt.figure()
    plt.hist(brightness_values, bins = 60, color = 'grey')
    plt.xlabel("Average Brightness Value")
    plt.ylabel("Number of Frames")
    plt.title(f"Histogram of Average Brightness - {input_path}")
    plt.xlim([0, 256])
    plt.grid(False)
    plt.show()

    print(f"\nProcessing complete. Output saved to {output_path}")

if __name__ == "__main__":
    input_video = "singapore.mp4"
    output_video = "processed_video.avi"
    talking_video = "talking.mp4"
    watermark1_img = "watermark1.png"
    watermark2_img = "watermark2.png"
    end_screen_video = "endscreen.mp4"

    process_video(input_video, output_video, talking_video, watermark1_img, watermark2_img, end_screen_video)