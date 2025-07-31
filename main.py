import cv2
import numpy as np
from matplotlib import pyplot as pt

# List of image files to process 
img_files = ["001.png", "002.png", "003.png", "004.png", "005.png", "006.png", "007.png", "008.png"]

# Threshold constants for line, gap and column detection 
LINE_THRESHOLD = 0.02
GAP_THRESHOLD = 1.8
COLUMN_THRESHOLD = 0.05

# Plot vertical and horizontal histograms to visualize pixel density
def plot_histograms(binary_img, img_file): 
    column_pixel_sums = np.sum(binary_img, axis = 0)
    row_pixel_sums = np.sum(binary_img, axis = 1)
    pt.figure()
    
    pt.subplot(1, 2, 1)
    pt.title(f"Vertical Histogram\n{img_file}")
    pt.xlabel("Column Number")
    pt.ylabel("Count")
    pt.xlim([0, len(column_pixel_sums)])
    pt.ylim([0, np.max(column_pixel_sums) * 1.1])
    pt.plot(column_pixel_sums)
    
    pt.subplot(1, 2, 2)
    pt.title(f"Horizontal Histogram\n{img_file}")
    pt.barh(range(len(row_pixel_sums)), row_pixel_sums, height=1.0)
    pt.xlabel("Count")
    pt.ylabel("Row Number")
    pt.xlim([0, np.max(row_pixel_sums) * 1.1]) 
    pt.ylim([0, len(row_pixel_sums)])
    pt.plot(row_pixel_sums)
    
    pt.show()

# Detect lines of text in a binary column image using horizontal projection
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

# Calculate the minimum gap between lines to consider as paragraph separation 
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
    min_gap = avg_gap * GAP_THRESHOLD # paragraph gap is larger than average line gap
    return min_gap

# Group text lines into paragraphs using the minimum gap threshold
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
    
# Detect columns in the document by analyzing vertical pixel density
def detect_columns(binary, threshold_ratio = COLUMN_THRESHOLD): 
    column_pixel_sums = np.sum(binary, axis = 0)
    col_threshold = np.max(column_pixel_sums) * threshold_ratio
    column_bounds = []
    inside_column = False
    x = 0

    for value in column_pixel_sums:
        if value > col_threshold and not inside_column:
            start_x = x # column starts 
            inside_column = True
        elif value <= col_threshold and inside_column:
            column_bounds.append((start_x, x)) # column ends
            inside_column = False
        x += 1
    return column_bounds
    
# Extract and save individual paragraph images from each column
def save_paragraphs(image_name, binary, column_bounds, original_img):
    count = 0

    for col_index in range(len(column_bounds)):
        x1, x2 = column_bounds[col_index]
        column_img = binary[:, x1:x2] # extract column
        
        lines = detect_lines(column_img, LINE_THRESHOLD)
        min_gap = calculate_min_gap(lines, GAP_THRESHOLD)
        paragraphs = group_lines_into_paragraphs(lines, min_gap)

        for para_index in range(len(paragraphs)):
            y1, y2 = paragraphs[para_index]
            paragraph_img = column_img[y1:y2, :]
            h, w = paragraph_img.shape

            if h / w > 4.0 and w < 100:
                continue

            filename = f"{image_name[:-4]}_c{col_index+1}_p{para_index+1}.png"
            color_paragraph = original_img[y1:y2, x1:x2]
            cv2.imwrite(filename, color_paragraph)

            count += 1 
    return count

# Process a single image file to extract paragraphs
def process_image(image_name):
    try:
        original_image = cv2.imread(image_name)
        if original_image is None:
            return 0
        
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
        plot_histograms(binary_image, image_name)

        column_bounds = detect_columns(binary_image)
        return save_paragraphs(image_name, binary_image, column_bounds, original_image)
    
    except Exception as e: 
        print(f"Error: {str(e)}")
        return 0
    
# Main entry point: Process all images and display extraction summary
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

# Main Function: Only execute the if this script is run directly
if __name__ == "__main__":
    main()