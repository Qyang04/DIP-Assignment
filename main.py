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