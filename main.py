
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === Step 1: Load and preprocess image ===
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 7))
    morph = cv2.dilate(binary, kernel, iterations=1)
    return img, binary, morph

# === Step 2: Detect and group lines into paragraph blocks ===
def extract_blocks(img, morph):
    height, width = morph.shape
    # Step 1: detect columns using vertical projection
    col_sum = np.sum(morph, axis=0)
    min_col_val = 300
    in_col = False
    col_start = 0
    columns = []
    for j, val in enumerate(col_sum):
        if val > min_col_val and not in_col:
            in_col = True
            col_start = j
        elif val <= min_col_val and in_col:
            col_end = j
            in_col = False
            if col_end - col_start > 50:
                columns.append((col_start, col_end))
    if not columns:
        columns = [(0, width)]

    # Step 2: process each column independently
    blocks = []
    for c_start, c_end in columns:
        col_img = img[:, c_start:c_end]
        col_morph = morph[:, c_start:c_end]
        contours, _ = cv2.findContours(col_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 50 and h > 10:
                line_boxes.append((x, y, w, h))

        line_boxes = sorted(line_boxes, key=lambda b: b[1])
        current_group = [line_boxes[0]]
        for i in range(1, len(line_boxes)):
            prev = current_group[-1]
            curr = line_boxes[i]
            vertical_gap = curr[1] - (prev[1] + prev[3])
            if vertical_gap < 40:
                current_group.append(curr)
            else:
                # Merge to one paragraph block
                x_vals = [b[0] for b in current_group]
                y_vals = [b[1] for b in current_group]
                w_vals = [b[0] + b[2] for b in current_group]
                h_vals = [b[1] + b[3] for b in current_group]
                x, y = min(x_vals), min(y_vals)
                w, h = max(w_vals) - x, max(h_vals) - y
                blocks.append({
                    'x': x + c_start,
                    'y': y,
                    'w': w,
                    'h': h,
                    'img': img[y:y+h, x + c_start:x + c_start + w]
                })
                current_group = [curr]

        # Final group
        if current_group:
            x_vals = [b[0] for b in current_group]
            y_vals = [b[1] for b in current_group]
            w_vals = [b[0] + b[2] for b in current_group]
            h_vals = [b[1] + b[3] for b in current_group]
            x, y = min(x_vals), min(y_vals)
            w, h = max(w_vals) - x, max(h_vals) - y
            blocks.append({
                'x': x + c_start,
                'y': y,
                'w': w,
                'h': h,
                'img': img[y:y+h, x + c_start:x + c_start + w]
            })

    # Sort column-wise top to bottom
    return sorted(blocks, key=lambda b: (b['x'] // 500, b['y']))

# === Step 3: Classify blocks as table or paragraph ===
def classify_blocks(blocks):
    paragraphs = []
    tables = []
    for block in blocks:
        w, h = block['w'], block['h']
        aspect_ratio = w / h
        area = w * h

        # Heuristics
        is_table = (
            aspect_ratio > 5 and h < 200  # short & wide
        ) or (
            h < 100 and w > 800           # flat and full width
        ) or (
            area > 150000 and aspect_ratio > 4  # large and very wide
        )

        if is_table:
            tables.append(block)
        else:
            paragraphs.append(block)
    return paragraphs, tables

# === Step 4: Save and show blocks ===
def save_and_show(blocks, prefix):
    for i, b in enumerate(blocks):
        filename = f'{prefix}_{i+1}.png'
        cv2.imwrite(filename, b['img'])
        plt.figure(figsize=(8, 2))
        plt.imshow(b['img'], cmap='gray')
        plt.title(f'{prefix.capitalize()} {i+1}')
        plt.axis('off')
        plt.show()
        
def draw_block_overlays(img, paragraphs, tables):
    color_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    for b in paragraphs:
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green for paragraphs
        cv2.putText(color_img, 'P', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for b in tables:
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # red for tables
        cv2.putText(color_img, 'T', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    plt.figure(figsize=(10, 12))
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.title("Paragraphs (Green) & Tables (Red)")
    plt.axis("off")
    plt.show()

# === Run on any image ===
def process_image(image_path):
    print(f"\n📄 Processing: {image_path}")
    img, binary, morph = preprocess_image(image_path)
    blocks = extract_blocks(img, morph)
    paragraphs, tables = classify_blocks(blocks)

    print(f"✅ Found {len(paragraphs)} paragraphs, {len(tables)} tables")
    save_and_show(paragraphs, 'paragraph')
    save_and_show(tables, 'table')
    
    #draw_block_overlays(img, paragraphs, tables)

# === Batch process ===
image_files = [f for f in os.listdir() if f.endswith('.png') and f[:3].isdigit()]
for img_file in sorted(image_files):
    process_image(img_file)