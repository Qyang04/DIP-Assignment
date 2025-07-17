# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:39:13 2025

@author: Sia Jia Le
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# List of image files to process 
img_files = ["Converted Paper (8)/001.png", "Converted Paper (8)/002.png", "Converted Paper (8)/003.png", "Converted Paper (8)/004.png", "Converted Paper (8)/005.png", "Converted Paper (8)/006.png", "Converted Paper (8)/007.png", "Converted Paper (8)/008.png"]

# Threshold constants for line, gap & column detection 
LINE_THRESHOLD = 0.02
GAP_THRESHOLD = 1.8
COLUMN_THRESHOLD = 0.05
MARGIN_SIZE = 40  # Margin size in pixels

# Plot vertical & horizontal histograms
def plot_histograms(binary_img, img_file): 
    
    # Sum black pixels along columns (vertical projection)
    column_pixel_sums = np.sum(binary_img, axis = 0)
    
    # Sum black pixels along rows (horizontal projection)
    row_pixel_sums = np.sum(binary_img, axis = 1)
    
    plt.figure()
    plt.subplots_adjust(wspace = 0.4)
    
    # Vertical Histogram (column detection)
    plt.subplot(1, 2, 1)
    plt.title(f"Vertical Histogram\n{img_file}")
    plt.xlabel("Column Number")
    plt.ylabel("Count")
    plt.xlim([0, len(column_pixel_sums)])
    plt.ylim([0, np.max(column_pixel_sums) * 1.1])
    plt.plot(column_pixel_sums)
    
    # Horizontal Histogram (row detection)
    plt.subplot(1, 2, 2)
    plt.title(f"Horizontal Histogram\n{img_file}")
    plt.barh(range(len(row_pixel_sums)), row_pixel_sums, height=1.0)
    plt.xlabel("Count")
    plt.ylabel("Row Number")
    plt.xlim([0, np.max(row_pixel_sums) * 1.1])  # Auto-scale with padding
    plt.ylim([0, len(row_pixel_sums)])
    plt.plot(row_pixel_sums)
    
    plt.show()
    
def detect_lines(column_img, LINE_THRESHOLD): 
    row_pixel_sums = np.sum(column_img, axis=1)
    threshold = np.max(row_pixel_sums) * LINE_THRESHOLD
    line_ranges = []
    line_start = None
    row_index = 0
    for val in row_pixel_sums:
        if val > threshold and line_start is None:
            line_start = row_index
        elif val <= threshold and line_start is not None: 
            line_ranges.append((line_start, row_index))
            line_start = None
        row_index += 1

    # Add the last line if image ends with text 
    if line_start is not None: 
        line_ranges.append((line_start, len(row_pixel_sums)))

    return line_ranges

# Calculate the minimum gap between lines to consider as paragraph separation 
def calculate_min_gap(lines, GAP_THRESHOLD): 
    line_gaps = []
    
    # Calculate gaps between consecutive lines
    for i in range(1, len(lines)):
        current_start = lines[i][0]
        previous_end = lines[i-1][1]
        gap = current_start - previous_end
        line_gaps.append(gap)

    if not line_gaps:
        return 0

    avg_gap = sum(line_gaps) / len(line_gaps)
    min_gap = avg_gap * GAP_THRESHOLD # paragraph gap is larger than average line gap

    return min_gap

# Group lines into paragraphs based on gap distances
def group_lines_into_paragraphs(lines, min_gap): 
    paragraphs = []
    current_paragraph = [lines[0]]

    for i in range(1, len(lines)): 
        gap = lines[i][0] - lines[i-1][1]
        # same paragraph
        if gap < min_gap:
            current_paragraph.append(lines[i]) 
        # end current paragraph and start new one
        else:
            paragraphs.append((current_paragraph[0][0], current_paragraph[-1][1]))
            current_paragraph = [lines[i]]

    # Add the last paragraph
    if current_paragraph:
        paragraphs.append((current_paragraph[0][0], current_paragraph[-1][1]))

    return paragraphs
    
# Detect columns in the document by analyzing vertical pixel density
def detect_columns(binary, threshold_ratio = COLUMN_THRESHOLD): 
    column_pixel_sums = np.sum(binary, axis = 0)
    col_threshold = np.max(column_pixel_sums) * threshold_ratio
    
    column_bounds = []
    inside_column = False
    
    x = 0

    # Find blocks of columns that exceeds the threshold 
    for value in column_pixel_sums:
        if value > col_threshold and not inside_column:
            start_x = x # column starts 
            inside_column = True
        elif value <= col_threshold and inside_column:
            column_bounds.append((start_x, x)) # column ends
            inside_column = False
        x += 1 
    if inside_column:
        column_bounds.append((start_x, x))
    return column_bounds
    
def detect_full_width_objects(binary_img, min_height = 30, min_width_ratio = 0.7, max_top = 250):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = binary_img.shape
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if h_box > min_height and w_box > min_width_ratio * w and y < max_top:
            boxes.append((y, y+h_box, x, x+w_box))
    boxes.sort(key = lambda b: b[0])
    return boxes

def save_with_margin(image, filename, margin=MARGIN_SIZE):
    image_with_margin = cv2.copyMakeBorder(
        image,
        margin, margin, margin, margin,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )
    cv2.imwrite(filename, image_with_margin)

# Extract and sace individual paragraphs from each column
def save_paragraphs(image_name, binary, column_bounds, original_img):
    h, w = binary.shape
    
    # Step 1: Detect full-width objects
    table_boxes = detect_full_width_objects(binary,  min_height = 30, min_width_ratio = 0.7, max_top = int(0.2 * h))
    occupied_rows = set()
    count = 1 
    base_name = os.path.splitext(os.path.basename(image_name))[0]

    # Create subfolder inside "outputs_images" for this image
    output_folder = os.path.join("outputs_images", base_name)
    os.makedirs(output_folder, exist_ok=True)  # Ensure output subfolder exists
    
    if table_boxes:
        for (y1, y2, x1, x2) in table_boxes:
            color_paragraph = original_img[y1:y2, x1:x2]
            filename = f"{base_name}_p{count}.png"
            output_image = os.path.join(output_folder, filename)
            save_with_margin(color_paragraph, output_image)
            occupied_rows.update(range(y1, y2))
            count += 1 
        
    # Step 2: Process columns for the rest
    for col_index in range(len(column_bounds)):
        x1, x2 = column_bounds[col_index]
        column_img = binary[:, x1:x2] # Extract column
        
        lines = detect_lines(column_img, LINE_THRESHOLD)
        if not lines:
            continue
        min_gap = calculate_min_gap(lines, GAP_THRESHOLD)
        paragraphs = group_lines_into_paragraphs(lines, min_gap)

        for para_index in range(len(paragraphs)):
            y1, y2 = paragraphs[para_index]
            if any(y in occupied_rows for y in range(y1, y2)):
                continue
            paragraph_img = column_img[y1:y2, :]
            h_p, w_p = paragraph_img.shape
            
            # Skip very tall, thin objects (often image artifacts)
            if h_p / w_p > 4.0 and w_p < 100:
                continue

            # Generate file name and save the paragraph
            # Generate file name and save the paragraph
            filename = f"{base_name}_p{count}.png"
            output_image = os.path.join(output_folder, filename)
            color_paragraph = original_img[y1:y2, x1:x2]
            save_with_margin(color_paragraph, output_image)

            count += 1
            
    return count-1

# Process a single image file to extract paragraphs
def process_image(image_name):
    try:
        # Read the original colour image
        original_image = cv2.imread(image_name)
        if original_image is None:
            print(f"Failed to read image: {image_name}")
            return 0
    
        # Convert to grayscale for processing
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
        # Binarize using Otsu's method (automatic thresholding)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
        # Plot histogram for visualization
        plot_histograms(binary_image, image_name)
    
        # Detect columns and save paragraphs 
        column_bounds = detect_columns(binary_image)
        return save_paragraphs(image_name, binary_image, column_bounds, original_image)
    
    except Exception as e: 
        print(f"Error: {str(e)}")
        return 0
    
# Main function to process all images and display summary 
def main():
    total_files = 0
    summary = []
    
    # process each image file
    for img_file in img_files:
        count = process_image(img_file)
        summary.append((img_file, count))
        total_files += count

    # Print summary of extraction
    print("==============================================")
    print("Extraction Summary:")
    print("==============================================")
    
    for filename, count in summary:
        print(f"{filename}: {count} paragraphs")
        
    print("\n==============================================")    
    print(f"Total paragraphs extracted: {total_files}")
    print("==============================================") 

if __name__ == "__main__":
    main()