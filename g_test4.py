import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output
import pandas as pd
import re
from skimage.morphology import skeletonize
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import argparse

# Global variables to store state
contours = []
outer_contours = []
first_level_contours = []
filtered_contours = []
outer_to_inner = {}
original_image = None
history = []
filtered_contours_data = {}
contour_data = {}
outer_contour_data = {}
first_level_contours_data = {}
final_contours = []

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\naush\Tesseract-OCR\Tesseract-OCR\tesseract.exe'


def detect_contours(image):
    global outer_contours, first_level_contours  # Declare globals to be modified within the function
    
    # Convert image to RGB for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to grayscale and then to binary
    original_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    binary_image = cv2.adaptiveThreshold(original_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)
    
    # Detect contours and hierarchy
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print("all contours: ", len(contours))
    #plot_contours(image_rgb, contours)

    # Initialize dictionaries for inner contours by parent index
    inner_contours_by_parent = {}
    children_of_first_level = set()

    outer_contours = []  # Initialize as an empty list
    outer_to_inner = {}  # Initialize as an empty dictionary
    first_level_contours = []  # Initialize as an empty list

    for i, contour in enumerate(contours):
        level = hierarchy[0][i][3]  # parent index
        if level == -1:  # No parent, thus an outer contour
            outer_contours.append(contour)
            outer_to_inner[i] = {'contour': contour, 'inner_contours': []}
            inner_contours_by_parent[i] = []
        else:
            parent_index = level
            if parent_index in outer_to_inner:
                outer_to_inner[parent_index]['inner_contours'].append(contour)
                inner_contours_by_parent.setdefault(parent_index, []).append(contour)
                children_of_first_level.add(i)

    # Determine first-level inner contours
    for parent_index, inner_contours in inner_contours_by_parent.items():
        if outer_to_inner[parent_index]['contour'] is not None:  # The parent is an outer contour
            if len(inner_contours) > 0:  # Check if the outer contour has any inner contours
                for contour in inner_contours:
                    first_level_contours.append(contour)

    image_area = original_image.shape[0] * original_image.shape[1]
    outer_contours = [c for c in outer_contours if cv2.contourArea(c) >= 0.00005 * image_area]
    first_level_contours = [c for c in first_level_contours if cv2.contourArea(c) >= 0.00005 * image_area]

    # Plot outer contours
    plot_outer_contours(image_rgb, outer_contours)

    # Plot first-level inner contours
    plot_first_level_inner_contours(image_rgb, first_level_contours)

    print("outer_contours: ", len(outer_contours))
    print("first_level_contours: ", len(first_level_contours))

    # Filter outer contours based on their associated inner contours
    filtered_outer_contours = []
    for outer_index, data in outer_to_inner.items():
        outer_contour = data['contour']
        inner_contours = data['inner_contours']
        
        if outer_contour is not None:
            outer_area = cv2.contourArea(outer_contour)
            total_inner_area = sum(cv2.contourArea(c) for c in inner_contours)
            
            if total_inner_area >= 0.85 * outer_area:
                filtered_outer_contours.append(outer_contour)
    
    # Create a new variable for filtered contours
    filtered_contours = first_level_contours + filtered_outer_contours
    filtered_contours = [c for c in filtered_contours if cv2.contourArea(c) >= 0.00005 * image_area]
    
    # Plot filtered contours
    #plot_filtered_contours(image_rgb, filtered_contours)
    
    print("filtered_contours: ", len(filtered_contours))

    # Store final contour data
    filtered_contours_data = {
        hash_contour(c): {'area': cv2.contourArea(c), 'moments': cv2.HuMoments(cv2.moments(c)).flatten()}
        for c in filtered_contours
    }
    
    outer_contour_data = {
        hash_contour(c): {'area': cv2.contourArea(c), 'moments': cv2.HuMoments(cv2.moments(c)).flatten()}
        for c in outer_contours
    }

    first_level_contours_data = {
        hash_contour(c): {'area': cv2.contourArea(c), 'moments': cv2.HuMoments(cv2.moments(c)).flatten()}
        for c in first_level_contours 
    }

    return outer_contour_data, first_level_contours_data, filtered_contours_data

"""
def plot_contours(image_rgb, contours):
    plt.figure(figsize=(10, 10))
    image_rgb_outer = image_rgb.copy()
    cv2.drawContours(image_rgb_outer, contours, -1, (0, 255, 0), 3)
    plt.imshow(image_rgb_outer)
    plt.title("ALL Contours")
    plt.axis('off')
    plt.show()

"""
def plot_outer_contours(image_rgb, outer_contours):
    plt.figure(figsize=(10, 10))
    image_rgb_outer = image_rgb.copy()
    cv2.drawContours(image_rgb_outer, outer_contours, -1, (0, 255, 0), 3)
    plt.imshow(image_rgb_outer)
    plt.title("Outer Contours")
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    contour_image = np.ones_like(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)) * 255
    cv2.drawContours(contour_image, outer_contours, -1, 0, thickness=cv2.FILLED)
    plt.imshow(contour_image, cmap='gray')
    plt.title("Contour Image")
    plt.axis('off')
    plt.show()

def plot_first_level_inner_contours(image_rgb, first_level_inner_contours):
    plt.figure(figsize=(10, 10))
    image_rgb_inner = image_rgb.copy()
    cv2.drawContours(image_rgb_inner, first_level_inner_contours, -1, (0, 0, 255), 3)
    plt.imshow(image_rgb_inner)
    plt.title("First-Level Inner Contours")
    plt.axis('off')
    plt.show()

"""
def plot_filtered_contours(image_rgb, filtered_contours):
    plt.figure(figsize=(10, 10))
    image_rgb_filtered = image_rgb.copy()
    cv2.drawContours(image_rgb_filtered, filtered_contours, -1, (0, 255, 0), 3)
    plt.imshow(image_rgb_filtered)
    plt.title("Filtered Contours")
    plt.axis('off')
    plt.show()

"""
def hash_contour(contour):
    return hash(tuple(map(tuple, contour.reshape(-1, 2))))


def extract_text_from_contour(image):
    ocr_result = pytesseract.image_to_data(image, output_type=Output.DICT, lang='train+eng+osd')
    text = []
    confidences = []

    for i in range(len(ocr_result['text'])):
        if int(ocr_result['conf'][i]) > 0:  # Only consider confident text results
            text.append(ocr_result['text'][i])
            confidences.append(float(ocr_result['conf'][i]))

    if text:
        confidence = confidences
        extracted_text = " ".join(text).strip()
    else:
        extracted_text = 'N/A'
        confidence = [0.0]

    return extracted_text, confidence

def process_image(image):
    #global final_contours
    global first_level_contours

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)

    shop_data = []
    contours_with_data = []
    skeleton_contours_with_data=[]

    for contour in first_level_contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = binary[y:y+h, x:x+w]

        extracted_text, confidence = extract_text_from_contour(roi)
        cleaned_shop_info = extracted_text

        shop_data.append({
            'name': cleaned_shop_info,
            'coord': (x, y, w, h),  # Using bounding box coordinates
            'confidence': confidence
        })
        contours_with_data.append({
            'contour': contour,
            'coord': (x, y, w, h),
            'name': cleaned_shop_info,
            'confidence': confidence
        })
    
    for contour in outer_contours:
        x, y, w, h = cv2.boundingRect(contour)

        skeleton_contours_with_data.append({
            'contour': contour,
            'coord': (x, y, w, h)

        })


    return contours_with_data, shop_data,skeleton_contours_with_data

def skeletonize_image(image, skeleton_contours_with_data):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contour_image = np.ones_like(gray) * 255
    for contour_info in skeleton_contours_with_data:
        cv2.drawContours(contour_image, [contour_info['contour']], -1, 0, thickness=cv2.FILLED)
        #cv2.drawContours(contour_image, [contour_info['contour']], -1, 0, 6)
        

    skeleton = skeletonize(contour_image == 255)
    path_points = np.column_stack(np.where(skeleton))

    return path_points

def highlight_and_find_max_distance_center_points(contours_with_data, updated_path_points):
    shop_points = []
    shop_data = []
    shop_names = {}

    def get_contour_side_centers(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return [
            (x + w, y + h // 2),  # Right center
            (x + w // 2, y + h),  # Bottom center
            (x, y + h // 2),      # Left center
            (x + w // 2, y)       # Top center
        ]

    for contour_info in contours_with_data:
        contour = contour_info['contour']
        centers = get_contour_side_centers(contour)

        # Find the closest center to any path point
        closest_center = None
        closest_distance = float('inf')

        for center in centers:
            distances = np.linalg.norm(updated_path_points - np.array(center[::-1]), axis=1)
            min_distance = np.min(distances)
            if min_distance < closest_distance:
                closest_distance = min_distance
                closest_center = center[::-1]

        if closest_center is not None:
            shop_points.append(closest_center)
            shop_data.append({
                'shop_name': contour_info.get('name', 'Unknown'),
                'shop_coord': tuple(closest_center),  # Ensure this is a tuple
                'confidence': contour_info.get('confidence', 0)
            })
            shop_names[contour_info.get('name', 'Unknown')] = tuple(closest_center)

    return shop_points, shop_data, shop_names

def filter_end_path_points(path_points, image_shape, margin=100):
    filtered_path_points = []
    remaining_path_points = []
    for point in path_points:
        x, y = point
        if x < margin or y < margin or x > image_shape[0] - margin or y > image_shape[1] - margin:
            filtered_path_points.append(point)
        else:
            remaining_path_points.append(point)
    return np.array(filtered_path_points), np.array(remaining_path_points)

def reduce_path_points(path_points, n_clusters=20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(path_points)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers

def plot_points(image, path_points, shop_points, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.scatter(path_points[:, 1], path_points[:, 0], c='red', s=1, label='Path Points')
    for sp in shop_points:
        plt.scatter(sp[1], sp[0], c='blue', s=30)
    plt.legend()
   
    plt.title(title)
    plt.show()

def calculate_shop_distance_matrix(shop_data):
    shop_names = [shop['shop_name'] for shop in shop_data]
    coordinates = np.array([shop['shop_coord'] for shop in shop_data])
    dist_matrix = distance_matrix(coordinates, coordinates)
    dist_df = pd.DataFrame(dist_matrix, index=shop_names, columns=shop_names)
    return dist_df

def main():
    global final_contours, current_contours, history

    # Path to the input image
    image_path = r'C:\Users\naush\Downloads\path\SHOW.jpg'
    image = load_image(image_path)

    detect_contours(image)

    # Extract shop data
    contours_with_data, shop_data,skeleton_contours_with_data = process_image(image)

    # Filter out shop data with alphanumeric names and non-blank names
    shop_data = [shop for shop in shop_data if shop['name'].strip() and re.search(r'[A-Za-z0-9]', shop['name'])]

    # Skeletonize and get path points
    path_points = skeletonize_image(image, skeleton_contours_with_data)

    # Filter end path points
    filtered_path_points, remaining_path_points = filter_end_path_points(path_points, image.shape)

    # Updated path points
    updated_path_points = remaining_path_points

    # Reduce path points
    final_path_points = reduce_path_points(updated_path_points, n_clusters=len(updated_path_points) // 20)

    # Plot the original path points
    plot_points(image, path_points, [], "Original Path Points")

    # Plot the filtered path points
    plot_points(image, filtered_path_points, [], "Filtered End Path Points")

    # Find shop points using updated path points
    shop_points, detailed_shop_data, shop_names = highlight_and_find_max_distance_center_points(contours_with_data, updated_path_points)

    # Filter out detailed shop data with alphanumeric names and non-blank names
    detailed_shop_data = [shop for shop in detailed_shop_data if shop['shop_name'].strip() and re.search(r'[A-Za-z0-9]', shop['shop_name'])]

    # Convert lists to tuples in detailed_shop_data
    for shop in detailed_shop_data:
        shop['shop_coord'] = tuple(shop['shop_coord'])

    # Convert to DataFrame and ensure no lists remain
    shop_df = pd.DataFrame(detailed_shop_data)
    shop_df = shop_df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)  # Convert lists to tuples
    shop_df = shop_df.drop_duplicates()

    # Plot the final path points and shop points
    plot_points(image, final_path_points, shop_points, "Final Path Points and Shop Points")

    # Save final path points to Excel
    path_df = pd.DataFrame(final_path_points, columns=['x', 'y'])
    path_df.to_excel(r'C:\projects\super_admin\final_path_points2.xlsx', index=False)

    # Save shop data to Excel
    shop_df.to_excel(r'C:\projects\super_admin\cordshop1.xlsx', index=False)

    # Calculate and save the shop distance matrix
    shop_distance_df = calculate_shop_distance_matrix(detailed_shop_data).drop_duplicates()
    shop_distance_df.to_excel(r'C:\projects\super_admin\shop_distance_matrix3.xlsx')

    print("Data saved successfully.")


    parser = argparse.ArgumentParser(description='Run image processing stages.')
    parser.add_argument('stage', choices=[
        'detect_contours',
        'process_image',
        'skeletonize_image',
        'highlight_and_find_max_distance_center_points',
        'filter_end_path_points',
        'reduce_path_points',
        'calculate_shop_distance_matrix'
    ], help='Stage to run')

    args = parser.parse_args()

    if args.stage == 'detect_contours':
        # Add the logic to save contours data
        pass
    elif args.stage == 'process_image':
        # Add the logic to process image and save contours_with_data and shop_data
        pass
    elif args.stage == 'skeletonize_image':
        # Add the logic to skeletonize image and save path_points
        pass
    elif args.stage == 'highlight_and_find_max_distance_center_points':
        # Add the logic to highlight and find max distance center points and save shop_points and shop_data_detailed
        pass
    elif args.stage == 'filter_end_path_points':
        # Add the logic to filter end path points and save filtered_path_points
        pass
    elif args.stage == 'reduce_path_points':
        # Add the logic to reduce path points and save final_path_points
        pass
    elif args.stage == 'calculate_shop_distance_matrix':
        # Add the logic to calculate shop distance matrix and save shop_distance_matrix
        pass

if __name__ == "__main__":
    main()
