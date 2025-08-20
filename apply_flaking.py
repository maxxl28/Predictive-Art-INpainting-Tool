import os
import cv2
import numpy as np
import random

# Main function to apply a flaking (weathering/damage) effect to an image
def apply_flaking(img_path, intensity=0.5):
    # Load the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    height, width = img.shape[:2]

    # Create a blank image (same size as input) to store the "flaking" mask
    masked_layer = np.zeros_like(img, dtype=np.uint8)

    # Utility function to clamp values within a given range
    def clamp(value, min_val, max_val):
        return max(min_val, min(value, max_val))

    # Generate the number of clusters for each type (big, small, minor) based on intensity
    num_of_big_clusters = clamp(random.randint(
        int(5.67 * intensity + 0.21),
        int((5.67 * intensity + 0.21) + 0.33 * intensity)
    ), 1, 100)

    num_of_small_clusters = clamp(random.randint(
        int(12.84 * intensity + 1.1),
        int((12.84 * intensity + 1.1) + 0.33 * intensity)
    ), 1, 200)

    num_of_minor_clusters = clamp(random.randint(
        int(30.13 * intensity - 1.75),
        int((30.13 * intensity - 1.75) + 0.33 * intensity)
    ), 1, 300)

    # Debug output to see how many clusters were generated
    print(f"Clusters - Big: {num_of_big_clusters}, Small: {num_of_small_clusters}, Minor: {num_of_minor_clusters}")

    # Function to generate and draw random shapes (ellipses or rectangles) representing flakes
    def process_cluster(x_reach_pct_min, x_reach_pct_max, subshapes_min, subshapes_max, num_clusters):
        for _ in range(num_clusters):
            try:
                # Randomly pick a center point for the cluster
                x_coord = random.randint(0, width)
                y_coord = random.randint(0, height)
                max_dim = max(width, height)
                z = random.uniform(.5, 1.5)  # Zoom factor for reach

                # Compute how far flakes can spread from the center
                x_reach = int(clamp(random.uniform(x_reach_pct_min, x_reach_pct_max), 0.001, 0.5) * max_dim * z)
                y_reach = x_reach * random.uniform(.5, 1.5)

                # Clamp bounding box for where to draw shapes
                x_low = clamp(x_coord - x_reach, 0, width)
                x_high = clamp(x_coord + x_reach, 0, width)
                y_low = clamp(y_coord - y_reach, 0, height)
                y_high = clamp(y_coord + y_reach, 0, height)

                # Number of shapes to draw in this cluster
                number_of_subshapes = random.randint(subshapes_min, subshapes_max)

                for _ in range(number_of_subshapes):
                    shape_decision = random.randint(0, 1)  # 0 = ellipse, 1 = rectangle
                    scaling = clamp(random.uniform(0.1, 0.15), 0.01, 1.0)

                    if shape_decision == 0:
                        # Create ellipse
                        center = (
                            int(random.uniform(x_low, x_high)),
                            int(random.uniform(y_low, y_high))
                        )
                        area = scaling * x_reach * y_reach
                        if area <= 0:
                            continue

                        ellipsis_x_length = int(np.sqrt(scaling) * x_reach * random.uniform(0.75, 1.25))
                        if ellipsis_x_length <= 0:
                            ellipsis_x_length = 1

                        ellipsis_y_length = max(1, int(area / ellipsis_x_length))
                        axes = (ellipsis_x_length, ellipsis_y_length)

                        # Draw filled white ellipse on mask
                        cv2.ellipse(masked_layer, center, axes, random.randint(0, 180), 0, 360, (255, 255, 255), -1)
                    else:
                        # Create rotated rectangle
                        rect_scaling = clamp(scaling * 3, 0.1, 3.0)
                        center = (
                            int(random.uniform(x_low, x_high)),
                            int(random.uniform(y_low, y_high))
                        )
                        area = rect_scaling * x_reach * y_reach
                        if area <= 0:
                            continue

                        x_length = max(1, int(np.sqrt(rect_scaling) * x_reach))
                        y_length = max(1, int(area / x_length))

                        size = (x_length, y_length)
                        angle = random.randint(0, 180)

                        # Define and draw rotated rectangle
                        rotated_rect = (center, size, angle)
                        vertices = cv2.boxPoints(rotated_rect)
                        vertices = np.int0(vertices)
                        cv2.drawContours(masked_layer, [vertices], 0, (255, 255, 255), -1)
            except Exception as e:
                print(f"Error in cluster processing: {e}")
                continue

    # Apply the function to big, small, and minor clusters with their own parameters
    process_cluster(0.03, 0.07, 45, 60, num_of_big_clusters)
    process_cluster(0.012, 0.045, 30, 50, num_of_small_clusters)
    process_cluster(0.004, 0.015, 10, 17, num_of_minor_clusters)

    # Combine the original image with the white mask (flakes) using bitwise OR
    modified_img = cv2.bitwise_or(img, masked_layer)
    
    return masked_layer, modified_img


script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    mask, result = apply_flaking("image2.jpg", intensity=.8)
    
    # Display results
    cv2.imshow("Flakes Mask", mask)
    cv2.imshow("Modified Image", result)
    
    # Save to script directory
    mask_path = os.path.join(script_dir, "22mona_lisa_damage.png")
    result_path = os.path.join(script_dir, "22mona_lisa_result.jpg")
    
    if cv2.imwrite(mask_path, mask) and cv2.imwrite(result_path, result):
        print(f"Files saved successfully to:\n{mask_path}\n{result_path}")
    else:
        print("Failed to save files! Check file permissions.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {e}")