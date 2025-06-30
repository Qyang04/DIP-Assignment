import cv2
import numpy as np

vid = cv2.VideoCapture("singapore.mp4") # Read the video (singapore)
out = cv2.VideoWriter('processed_video.avi', # Set the file name of the new video
                      cv2.VideoWriter_fourcc(*'MJPG'), # Set the codec
                      30.0, # Set the frame rate
                      (1280, 720)) # Set the resolution (width, height)
total_no_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT) # Get total number of frames

frames = [] # List to store all frames of the video
brightness_values = [] # List to store brightness values for each frame

# Loop through all frames in the video
for frame_count in range(0, int(total_no_frames)): 
    success, frame = vid.read() # Read a single frame from the video
    if not success: # If reading failed, break the loop
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale
    brightness_values.append(np.mean(gray)) # Calculate and store the mean brightness
    frames.append(frame) # Store the frame in the list
    
# Detect if the video is taken during daytime or nighttime after analyzing all frames
average_brightness = np.mean(brightness_values)
night_threshold = 100 # Threshold value to decide if the video is nighttime
is_night = average_brightness < night_threshold # If the average brightness is less than threshold value, then it is nighttime
print(f"Average brightness of the video: {average_brightness:.2f} | Nightime: {is_night}")
    
# Brighten the video function
def brighten(frame, factor=1.6):
    # Increase brightness of all pixels by multiplying factor
    # Ensure pixel values stay within 0 (black) to 255 (white)
    # Convert result to 8-bit unsigned integer type
    return np.clip(frame * factor, 0, 255).astype(np.uint8)

# Loop through all stored frames to process and write them to output video
for frame in frames:
    if is_night: # If it is nightime, brighten the frame
        frame = brighten(frame)
    out.write(frame) # Save the processed frame in the new video
        
vid.release()
out.release()