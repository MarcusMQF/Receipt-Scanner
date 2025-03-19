import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import easyocr

def preprocess_image(image_bytes):
    """
    Preprocess the image for better OCR results
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Apply dilation to connect text components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.dilate(gray, kernel, iterations=1)
    
    # Encode the processed image back to bytes
    success, processed_image = cv2.imencode('.png', gray)
    return io.BytesIO(processed_image.tobytes())

def extract_text(image_bytes):
    """
    Extract text from the image using EasyOCR
    """
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Read image
    nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Extract text
    results = reader.readtext(image)
    
    # Combine all detected text
    text = '\n'.join([result[1] for result in results])
    
    return text

def format_text(text):
    """
    Format the extracted text for better readability
    
    Args:
        text: str, raw extracted text
        
    Returns:
        str: Formatted text
    """
    # Split text into lines
    lines = text.split('\n')
    
    # Remove empty lines and excessive whitespace
    lines = [line.strip() for line in lines if line.strip()]
    
    # Join lines back together
    formatted_text = '\n'.join(lines)
    
    return formatted_text 