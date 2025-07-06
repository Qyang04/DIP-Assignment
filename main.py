# main.py
import cv2
import numpy as np

# Function to resize and overlay the video
def resizeAndOverlayVideo(foregroundVideoPath, backgroundVideoPath):
    # Define a video object based on the video that we want to resize
    foregroundVideo = cv2.VideoCapture(foregroundVideoPath)
    backgroundVideo = cv2.VideoCapture(backgroundVideoPath)
    
    scale_percent = 30
    lastBackgroundFrame = None
    
    # Capture video frame by frame
    foregroundSuccess, foreground = foregroundVideo.read()
    backgroundSuccess, background = backgroundVideo.read()
        
    while foregroundSuccess or backgroundSuccess:

        # Handle background
        if backgroundSuccess:
            lastBackgroundFrame = background.copy()
        elif lastBackgroundFrame is not None:
            background = lastBackgroundFrame
        else:
            break #No valid background frame at all
        
        if foregroundSuccess:
            # Get the current frame size
            height, width, _ = foreground.shape
            # Resize the frame
            new_width = int(width*scale_percent/100)
            new_height = int(height*scale_percent/100)
            new_frame_size = (new_width, new_height)
            resizedForegroundVideo = cv2.resize(foreground, new_frame_size, interpolation = cv2.INTER_AREA)\
            
            # Resize background if necessary to match overlay
            bg_height, bg_width = background.shape[:2]
            if bg_height < new_height or bg_width < new_width:
                background = cv2.resize(background, (max(new_width, bg_width), max(new_height, bg_height)))
    
            # Overlay resized foreground directly at top-left corner
            background[0:new_height, 0:new_width] = resizedForegroundVideo

        # Show the result
        cv2.imshow("Resize and Overlay", background)
        
        # Read the next frames
        foregroundSuccess, foreground = foregroundVideo.read()
        backgroundSuccess, background = backgroundVideo.read()
        
        # Video will update every 30 milliseconds which simulates a frame rate of 33 frames per second
        cv2.waitKey(30) 
        
    foregroundVideo.release()
    backgroundVideo.release()
    cv2.destroyAllWindows()
    
resizeAndOverlayVideo(r"C:\Users\Sia Jia Le\OneDrive - Sunway Education Group\Digital Image Processing\Group Assignment\CSC2014- Group Assignment_Aug-2025\talking.mp4", r"C:\Users\Sia Jia Le\OneDrive - Sunway Education Group\Digital Image Processing\Group Assignment\CSC2014- Group Assignment_Aug-2025\Recorded Videos (4)\singapore.mp4")