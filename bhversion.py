import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# List of image file paths to process
img_files = ["Converted Paper (8)/001.png", "Converted Paper (8)/002.png", "Converted Paper (8)/003.png", "Converted Paper (8)/004.png", "Converted Paper (8)/005.png", "Converted Paper (8)/006.png", "Converted Paper (8)/007.png", "Converted Paper (8)/008.png"]

# Threshold values used for detecting lines, gaps between paragraphs, and columns
# Margin added around extracted paragraph images
LINE_THRESHOLD = 0.02
GAP_THRESHOLD = 1.8
COLUMN_THRESHOLD = 0.05
MARGIN_SIZE = 40 

# Plot vertical and horizontal histograms to visualize black pixel distribution
def plot_histograms(binary_img, img_file): 
    column_pixel_sums = np.sum(binary_img, axis = 0)
    row_pixel_sums = np.sum(binary_img, axis = 1)
    plt.figure()
    plt.subplots_adjust(wspace = 0.4)

    # Vertical histogram (columns)
    plt.subplot(1, 2, 1)
    plt.title(f"Vertical Histogram\n{img_file}")
    plt.xlabel("Column Number")
    plt.ylabel("Count")
    plt.xlim([0, len(column_pixel_sums)])
    plt.ylim([0, np.max(column_pixel_sums) * 1.1])
    plt.plot(column_pixel_sums)
    
    # Horizontal histogram (rows)
    plt.subplot(1, 2, 2)
    plt.title(f"Horizontal Histogram\n{img_file}")
    plt.barh(range(len(row_pixel_sums)), row_pixel_sums, height=1.0)
    plt.xlabel("Count")
    plt.ylabel("Row Number")
    plt.xlim([0, np.max(row_pixel_sums) * 1.1]) 
    plt.ylim([0, len(row_pixel_sums)])
    plt.plot(row_pixel_sums)
    
    plt.show()

# Detect the start and end of text lines in a binary image
# Use row-wise pixel density to find where lines begin and end
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

    if line_start is not None: 
        line_ranges.append((line_start, len(row_pixel_sums)))
    return line_ranges

# Calculate the minimum vertical gap required to separate paragraphs
# Based on average vertical distances between lines
def calculate_min_gap(lines, GAP_THRESHOLD): 
    line_gaps = []
    
    for i in range(1, len(lines)):
        current_start = lines[i][0]
        previous_end = lines[i-1][1]
        gap = current_start - previous_end
        line_gaps.append(gap)

    if not line_gaps:
        return 0

    avg_gap = sum(line_gaps) / len(line_gaps)
    min_gap = avg_gap * GAP_THRESHOLD
    return min_gap

# Group detected text lines into paragraphs using the calculated minimum gap distance
# If the gap between two lines is smaller than the minimum gap, they are same paragraph
def group_lines_into_paragraphs(lines, min_gap): 
    paragraphs = []
    current_paragraph = [lines[0]]

    for i in range(1, len(lines)): 
        gap = lines[i][0] - lines[i-1][1]

        if gap < min_gap:
            current_paragraph.append(lines[i]) 
        else:
            paragraphs.append((current_paragraph[0][0], current_paragraph[-1][1]))
            current_paragraph = [lines[i]]

    if current_paragraph:
        paragraphs.append((current_paragraph[0][0], current_paragraph[-1][1]))
    return paragraphs
    
# Detect vertical regions (columns) in the document by analyzing vertical pixel density
# Identify column start and end positions based on whether the pixel intensity exceeds a defined threshold
def detect_columns(binary, threshold_ratio = COLUMN_THRESHOLD): 
    column_pixel_sums = np.sum(binary, axis = 0)
    col_threshold = np.max(column_pixel_sums) * threshold_ratio
    column_bounds = []
    inside_column = False
    x = 0

    for value in column_pixel_sums:
        if value > col_threshold and not inside_column:
            start_x = x 
            inside_column = True
        elif value <= col_threshold and inside_column:
            column_bounds.append((start_x, x)) 
            inside_column = False
        x += 1 
    if inside_column:
        column_bounds.append((start_x, x))
    return column_bounds
    
# Detect full-width horizontal elements (like tables) that span nearly the full width of the page
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

# Save an image with a white margin around it for better visual separation
def save_with_margin(image, filename, margin=MARGIN_SIZE):
    image_with_margin = cv2.copyMakeBorder(
        image,
        margin, margin, margin, margin,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )
    cv2.imwrite(filename, image_with_margin)

# Extract and save individual paragraph images for a given image
def save_paragraphs(image_name, binary, column_bounds, original_img):
    h, w = binary.shape
    
    # Step 1: Detect full-width objects like table at top
    table_boxes = detect_full_width_objects(binary,  min_height = 30, min_width_ratio = 0.7, max_top = int(0.2 * h))
    occupied_rows = set()
    count = 1 
    base_name = os.path.splitext(os.path.basename(image_name))[0]

    output_folder = os.path.join("outputs_Task B", base_name)
    os.makedirs(output_folder, exist_ok=True)  
    
    if table_boxes:
        for (y1, y2, x1, x2) in table_boxes:
            color_paragraph = original_img[y1:y2, x1:x2]
            filename = f"{base_name}_p{count}.png"
            output_image = os.path.join(output_folder, filename)
            save_with_margin(color_paragraph, output_image)
            occupied_rows.update(range(y1, y2))
            count += 1 
        
    # Step 2: Process columns for text paragraphs, images, and tables
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

            if h_p / w_p > 4.0 and w_p < 100:
                continue

            filename = f"{base_name}_p{count}.png"
            output_image = os.path.join(output_folder, filename)
            color_paragraph = original_img[y1:y2, x1:x2]
            save_with_margin(color_paragraph, output_image)
            count += 1           
    return count-1

# Process a single image: convert to binary, visualize with histograms, detect layout, extract paragraphs, and save
def process_image(image_name):
    try:
        original_image = cv2.imread(image_name)
        if original_image is None:
            print(f"Failed to read image: {image_name}")
            return 0

        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
        plot_histograms(binary_image, image_name)
    
        column_bounds = detect_columns(binary_image)
        return save_paragraphs(image_name, binary_image, column_bounds, original_image)
    
    except Exception as e: 
        print(f"Error: {str(e)}")
        return 0
    
# Entry point of the script
# Main function: Process all input images, extract paragraphs, and display summary 
def main():
    total_files = 0
    summary = []
    
    for img_file in img_files:
        count = process_image(img_file)
        summary.append((img_file, count))
        total_files += count

    print("==============================================")
    print("Extraction Summary:")
    print("==============================================")
    
    for filename, count in summary:
        print(f"{filename}: {count} paragraphs")
        
    print("\n==============================================")    
    print(f"Total paragraphs extracted: {total_files}")
    print("==============================================") 

# Run the program
if __name__ == "__main__":
    main()