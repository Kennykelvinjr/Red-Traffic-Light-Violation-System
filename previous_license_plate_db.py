import os
import cv2
import numpy as np
import mysql.connector
from paddleocr import PaddleOCR
from datetime import datetime


def create_database():
    """Create MariaDB database with required tables if they don't exist"""
    # Connect to MariaDB
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  # Replace with your MariaDB username
        password="",  # Replace with your MariaDB password
        auth_plugin='mysql_native_password'
    )

    cursor = conn.cursor()

    # Create database if it doesn't exist
    cursor.execute("CREATE DATABASE IF NOT EXISTS license_plate")
    cursor.execute("USE license_plate")

    # Create table for license plate records
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS license_records (
        id INT AUTO_INCREMENT PRIMARY KEY,
        license_plate VARCHAR(50) NOT NULL,
        offense VARCHAR(100),
        timestamp DATETIME,
        image_path VARCHAR(255)
    )
    ''')

    conn.commit()
    return conn, cursor


# Initialize PaddleOCR
ocr = PaddleOCR()


def perform_ocr(image_array):
    """
    Perform OCR on an image array using PaddleOCR

    Args:
        image_array: Image array to perform OCR on

    Returns:
        Detected text as a string
    """
    if image_array is None:
        raise ValueError("Image is None")

    # Perform OCR on the image array
    results = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition
    detected_text = []

    # Process OCR results
    if results[0] is not None:
        for result in results[0]:
            text = result[1][0]
            detected_text.append(text)

    # Join all detected texts into a single string
    return ''.join(detected_text)


def detect_license_plate(image_path):
    """
    Detect and extract license plate text from an image using PaddleOCR

    Args:
        image_path: Path to the image file

    Returns:
        Detected license plate text or None if not detected
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply filtering to enhance text
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection
    edged = cv2.Canny(gray, 30, 200)

    # Find contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    license_plate_text = None

    # Loop through contours to find license plate
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        # If our contour has 4 points, it's likely a license plate
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            # Extract the license plate region
            plate_region = img[y:y + h, x:x + w]

            # Use PaddleOCR to extract text
            try:
                text = perform_ocr(plate_region)

                # Clean the text (remove non-alphanumeric characters)
                cleaned_text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').replace('-',
                                                                                                                ' ')

                if cleaned_text:
                    license_plate_text = cleaned_text
                    break
            except Exception as e:
                print(f"OCR error: {e}")
                continue

    # If no license plate was found with contour method, try direct OCR on the whole image
    if license_plate_text is None:
        try:
            text = perform_ocr(img)
            cleaned_text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').replace('-', ' ')
            if cleaned_text:
                license_plate_text = cleaned_text
        except Exception as e:
            print(f"OCR error on full image: {e}")

    return license_plate_text


def process_images_folder(folder_path, conn, cursor):
    """
    Process all images in the specified folder and add detected license plates to database

    Args:
        folder_path: Path to folder containing images
        conn: Database connection
        cursor: Database cursor
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(folder_path)
                   if os.path.isfile(os.path.join(folder_path, f)) and
                   os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {folder_path}")
        return

    print(f"Found {len(image_files)} images to process")

    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing image {i}/{len(image_files)}: {image_file}")

        license_plate = detect_license_plate(image_path)

        if license_plate:
            print(f"Detected license plate: {license_plate}")

            # Ask for offense type
            offense = input(f"Enter offense for license plate {license_plate} (or press Enter to skip): ")

            if offense:  # Only add to database if offense is provided
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Insert into database
                cursor.execute(
                    "INSERT INTO license_records (license_plate, offense, timestamp, image_path) VALUES (%s, %s, %s, %s)",
                    (license_plate, offense, timestamp, image_path)
                )
                conn.commit()
                print(f"Added to database with offense: {offense}")
            else:
                print("Skipped adding to database (no offense provided)")
        else:
            print(f"No license plate detected in {image_file}")


def process_single_image(image_path, offense="Red Light Violation", conn=None, cursor=None):
    """
    Process a single image and add detected license plate to database

    Args:
        image_path: Path to the image file
        offense: Type of offense (default is "Red Light Violation")
        conn: Optional database connection
        cursor: Optional database cursor

    Returns:
        Detected license plate text or None if not detected
    """
    # Create a new connection if one wasn't provided
    connection_created = False
    if conn is None or cursor is None:
        conn, cursor = create_database()
        connection_created = True

    # Detect license plate
    license_plate = detect_license_plate(image_path)

    if license_plate:
        print(f"Detected license plate: {license_plate}")

        # Add to database with the provided offense
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Insert into database
        cursor.execute(
            "INSERT INTO license_records (license_plate, offense, timestamp, image_path) VALUES (%s, %s, %s, %s)",
            (license_plate, offense, timestamp, image_path)
        )
        conn.commit()
        print(f"Added to database with offense: {offense}")
    else:
        print(f"No license plate detected in {image_path}")

    # Only close the connection if we created it
    if connection_created:
        conn.close()

    return license_plate


def main():
    # Create or connect to the database
    conn, cursor = create_database()

    # Get current date for the folder path
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Process images in the saved_images folder with current date
    images_folder = os.path.join("/home/adrian/Red-Traffic-Light-Violation/saved_images", current_date)
    if not os.path.exists(images_folder):
        print(f"Error: Folder {images_folder} does not exist")

        # List available date folders
        base_folder = "/home/adrian/Red-Traffic-Light-Violation/saved_images"
        if os.path.exists(base_folder):
            available_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
            if available_folders:
                print(f"Available date folders: {', '.join(available_folders)}")
                # Use the most recent folder
                latest_folder = sorted(available_folders)[-1]
                images_folder = os.path.join(base_folder, latest_folder)
                print(f"Using the most recent folder: {latest_folder}")
            else:
                print(f"No date folders found in {base_folder}")
                return
        else:
            print(f"Base folder {base_folder} does not exist")
            return

    print(f"Processing images from folder: {images_folder}")
    process_images_folder(images_folder, conn, cursor)

    # Display all records in the database
    cursor.execute("SELECT id, license_plate, offense, timestamp FROM license_records")
    records = cursor.fetchall()

    print("\nLicense Plate Database Records:")
    print("=" * 80)
    print(f"{'S/N':<5} {'License Plate':<20} {'Offense':<30} {'Timestamp':<25}")
    print("-" * 80)

    for record in records:
        print(f"{record[0]:<5} {record[1]:<20} {record[2]:<30} {str(record[3]):<25}")

    # Close the database connection
    conn.close()


if __name__ == "__main__":
    main()