# main.py
import cv2

# Read main video
vid = cv2.VideoCapture(r"C:\Users\Irenehaha\OneDrive - Sunway Education Group\DIP Ass1\CSC2014- Group Assignment_Aug-2025\Recorded Videos (4)\singapore.mp4")

# Read talking and end screen video
end_vid = cv2.VideoCapture(r"C:\Users\Irenehaha\OneDrive - Sunway Education Group\DIP Ass1\CSC2014- Group Assignment_Aug-2025\endscreen.mp4")


# Read watermark image using OpenCV with transparency
# cv2.IMREAD_UNCHANGED ask to load 4 color channels instead of the default 3 color channels
watermark1 = cv2.imread(r"C:\Users\Irenehaha\OneDrive - Sunway Education Group\DIP Ass1\CSC2014- Group Assignment_Aug-2025\watermark1.png")
watermark2 = cv2.imread(r"C:\Users\Irenehaha\OneDrive - Sunway Education Group\DIP Ass1\CSC2014- Group Assignment_Aug-2025\watermark2.png")

# === Function to add watermark without resizing through blending image===
# dst = src1 * alpha + src2 * beta + gamma -- output img = 1st input img * weight of 1st element + 2nd image  * weight of 2nd image + scalar value to each sum
# https://www.geeksforgeeks.org/python/addition-blending-images-using-opencv-python/
def add_watermark_full(frame, watermark):
     return cv2.addWeighted(frame, 1, watermark, 0.5, 0)


# Output settings: Set the file name of the new video, codec, frame rate, resolution(width,height)
out = cv2.VideoWriter('processed_singapore1.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      30.0,
                      (1280,720))

# Get the total number of frames
total_no_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

# Start processing main video: Loop through all the frames
for frame_count in range(0,int(total_no_frames)):
    success,frame = vid.read() # Read a single frame from the video
    if not success:
        break

    frame = cv2.resize(frame,(1280,720))

    # Alternate watermark every 4 seconds (4s * 30fps -- there is 120 frames)
    if (frame_count // 120) % 2 == 0:
        frame = add_watermark_full(frame, watermark1)
    else:
        frame = add_watermark_full(frame, watermark2)

    # Save processed frame into the new video
    out.write(frame)

# Add endscreen video
while True:
     success_end, end_frame = end_vid.read()
     if not success_end:
        break
     end_frame = cv2.resize(end_frame, (1280,720))
     out.write(end_frame)

# Release everything
vid.release()
end_vid.release()
out.release()
cv2.destroyAllWindows()



