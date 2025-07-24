import cv2
import numpy as np

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

# === Read video ===
vid = cv2.VideoCapture(r"C:\Users\Irenehaha\OneDrive - Sunway Education Group\DIP Ass1\CSC2014- Group Assignment_Aug-2025\Recorded Videos (4)\singapore.mp4")

# === Set output file ===
out = cv2.VideoWriter("processed_video.avi",
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      30.0,
                      (1280, 720))

# === Get total number of frames ===
total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# === Read watermark images ===
watermark1 = remove_black_background(r"C:\Users\Irenehaha\OneDrive - Sunway Education Group\DIP Ass1\CSC2014- Group Assignment_Aug-2025\watermark1.png")
watermark2 = remove_black_background(r"C:\Users\Irenehaha\OneDrive - Sunway Education Group\DIP Ass1\CSC2014- Group Assignment_Aug-2025\watermark2.png")

# === To loop through all the frames ===
for frame_count in range(total_no_frames):
    success, frame = vid.read()  # Read a single frame
    if not success:
        break

    frame = cv2.resize(frame, (1280, 720))  # Set resolution

    # === Alternate watermark every 4 seconds (4s * 30fps = 120 frames) ===
    if (frame_count // 120) % 2 == 0:
        frame = overlay_transparent(frame, watermark1)
    else:
        frame = overlay_transparent(frame, watermark2)

    out.write(frame)  # Save processed frame

# === Add the end screen video to the end of the video ===
add_endscreen(out,
              r"C:\Users\Irenehaha\OneDrive - Sunway Education Group\DIP Ass1\CSC2014- Group Assignment_Aug-2025\endscreen.mp4",
              1280, 720)

# === Release resources ===
vid.release()
out.release()
cv2.destroyAllWindows()