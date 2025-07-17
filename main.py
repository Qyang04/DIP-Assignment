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

"""
def overlay_talking_video(main_frame, talking_frame):
    # Resize talking video to 25% of main frame
    h, w = main_frame.shape[:2]
    talking_frame = cv2.resize(talking_frame, (w//4, h//4))
    
    # Overlay on top-left corner
    main_frame[0:h//4, 0:w//4] = talking_frame
    
    return main_frame

def add_watermark(frame, watermark1, watermark2):
    h, w = frame.shape[:2]
    
    # Resize watermarks
    watermark1 = cv2.resize(watermark1, (w//5, h//10))
    watermark2 = cv2.resize(watermark2, (w//5, h//10))
    
    # Overlay watermark1 (top-right)
    frame[0:h//10, w-w//5:w] = watermark1
    
    # Overlay watermark2 (bottom-left)
    frame[h-h//10:h, 0:w//5] = watermark2
    
    return frame

def append_end_screen(video_writer, end_screen_path):
    end_vid = cv2.VideoCapture(end_screen_path)
    while True:
        success, frame = end_vid.read()
        if not success:
            break
        video_writer.write(frame)
    end_vid.release()
"""

def process_video(input_path, output_path):
#def process_video(input_path, output_path, talking_path, watermark1_path, watermark2_path, end_screen_path):
    
    print(input_path)
    
    # Load videos and watermarks
    vid = cv2.VideoCapture(input_path)
#    talking_vid = cv2.VideoCapture(talking_path)
#    watermark1 = cv2.imread(watermark1_path, cv2.IMREAD_UNCHANGED)
#    watermark2 = cv2.imread(watermark2_path, cv2.IMREAD_UNCHANGED)
    
    # Get video properties
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_no_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    
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
    for frame_count in range(0, int(total_no_frames)):
        success, frame = vid.read()
        if not success:
            print(f"Stopped at frame {frame_count} (may be incomplete video)")
            break
        
        # If it is nightime, brighten the frame
        if is_night: 
            frame = brighten(frame)
        
        # Step 2: Face Blurring
        frame = blur_faces(frame)
        """
        # Step 3: Overlay Talking Video
        ret_talking, talking_frame = talking_vid.read()
        if ret_talking:
            frame = overlay_talking_video(frame, talking_frame)
        else:
            talking_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset if video ends
        
        # Step 4: Add Watermarks
        frame = add_watermark(frame, watermark1, watermark2)
        """
        # Write processed frame
        out.write(frame)
        
        if frame_count % 100 == 0 or frame_count == int(total_no_frames)-1:
            print(f"Processing frame {frame_count}/{int(total_no_frames)} | "
                  f"Brightness: {brightness_values[frame_count]:.1f}")
    
    # Step 5: Append End Screen
#    append_end_screen(out, end_screen_path)
    
    # Release resources
    vid.release()
#    talking_vid.release()
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
    input_video = "Recorded Videos (4)/singapore.mp4"
    output_video = "processed_video.avi"
#    talking_video = "talking.mp4"
#    watermark1_img = "watermark1.png"
#    watermark2_img = "watermark2.png"
#    end_screen_video = "endscreen.mp4"
    
    process_video(input_video, output_video)
#    process_video(input_video, output_video, talking_video, watermark1_img, watermark2_img, end_screen_video)
import cv2
import numpy as np

# Function to resize the foreground video and overlay it on top of the background video
def resizeAndOverlayVideo(foregroundVideoPath, backgroundVideoPath, outputPath):
    # Load the foreground and background videos
    foregroundVideo = cv2.VideoCapture(foregroundVideoPath)
    backgroundVideo = cv2.VideoCapture(backgroundVideoPath)
    
    # Scale percentage to shrink the foreground video
    scale_percent = 30
    lastBackgroundFrame = None
    
    # Capture video frame by frame
    foregroundSuccess, foreground = foregroundVideo.read()
    backgroundSuccess, background = backgroundVideo.read()
    
    # Get the frame width, height, and frames-per-second (fps) from the background video
    frame_width = int(backgroundVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(backgroundVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = backgroundVideo.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30  # fallback if FPS can't be read
    
    # Set up the video writer to save the output to the specified file path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outputPath, fourcc, fps, (frame_width, frame_height))
        
    # Process frames until both videos have ended
    while foregroundSuccess or backgroundSuccess:

        # Handle background:
        # If the background has a new frame, update the last valid frame
        if backgroundSuccess:
            lastBackgroundFrame = background.copy()
        # If background ends, reuse the last valid background frame
        elif lastBackgroundFrame is not None:
            background = lastBackgroundFrame
        else:
            break  # Exit if there's no background at all
        
        # If there is a foreground frame, resize and overlay it
        if foregroundSuccess:
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
        

        # Write the processed frame to the output video file
        out.write(background)
        
        # Display the current frame in a window (for visual feedback)
        cv2.imshow("Resize and Overlay", background)
        
        # Read the next frames from both videos
        foregroundSuccess, foreground = foregroundVideo.read()
        backgroundSuccess, background = backgroundVideo.read()
        
        # Wait 30ms between frames which simulates a frame rate of 33 frames per second
        cv2.waitKey(30) 
        
    # Release all resources after processing
    foregroundVideo.release()
    backgroundVideo.release()
    out.release()
    cv2.destroyAllWindows()
    
# Call the function with input and output file paths
resizeAndOverlayVideo(r"C:\Users\Sia Jia Le\OneDrive - Sunway Education Group\Digital Image Processing\Group Assignment\CSC2014- Group Assignment_Aug-2025\talking.mp4", r"C:\Users\Sia Jia Le\OneDrive - Sunway Education Group\Digital Image Processing\Group Assignment\CSC2014- Group Assignment_Aug-2025\Recorded Videos (4)\singapore.mp4", r"C:\Users\Sia Jia Le\OneDrive - Sunway Education Group\Digital Image Processing\Group Assignment\CSC2014- Group Assignment_Aug-2025\Recorded Videos (4)\Resize and Overlay.mp4")