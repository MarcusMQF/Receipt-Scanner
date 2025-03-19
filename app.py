import streamlit as st
from huggingface_hub import InferenceClient
import base64
from PIL import Image
import io
import easyocr
import re
import numpy as np
import cv2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Hugging Face client
client = InferenceClient(token=os.getenv("HUGGING_FACE_API_KEY"))

def extract_text_from_image(image_bytes):
    """Extract text from image using EasyOCR"""
    reader = easyocr.Reader(['en'])
    
    # Convert bytes to numpy array for EasyOCR
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Extract text
    results = reader.readtext(image)
    return [text[1] for text in results]

def analyze_receipt_with_model(image_bytes):
    """Analyze receipt using Hugging Face model"""
    # Convert image bytes to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # Prepare the messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this receipt image and format the output EXACTLY as follows with consistent spacing:

### Bill Details

- Date and time: [value]
- Table number: [value]
- Server: [value]
- Order number: [value]
- Invoice number: [value]


### Order Details

| Number | Item | Quantity | Price before tax | Price after tax | Total |
|--------|------|----------|------------------|-----------------|-------|
| 1 | [item name] | [qty] | RM [amount] | RM [amount] | RM [amount] |
[continue for all items...]


### Tax

| Type | Percentage | Amount |
|------|------------|--------|
| SST | [percentage] | RM [amount] |
[other taxes if any...]


### Total

- Subtotal: RM [amount]
- Tax: RM [amount]
- Total: RM [amount]

Note: Please ensure:
1. Each section header starts with '### '
2. There are TWO empty lines between each section
3. All currency values have 'RM' prefix
4. Tables are properly formatted with aligned columns
5. All numbers have 2 decimal places"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    # Call the Hugging Face model
    response = client.post(
        model="microsoft/git-base",  # You can change this to your preferred model
        json={"inputs": messages}
    )
    
    return response

# Page configuration
st.set_page_config(
    page_title="Receipt Analyzer",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.markdown("""
    # 🧾 Receipt Analyzer
    Extract and analyze receipt information using AI! Results are formatted in markdown for easy copying.
""")

# Add clear button
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear 🗑️"):
        if 'analysis_result' in st.session_state:
            del st.session_state['analysis_result']
        st.rerun()

st.markdown("---")

# Sidebar for upload
with st.sidebar:
    st.header("Upload Receipt")
    uploaded_file = st.file_uploader("Choose a receipt image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Create a container for the image and button
        container = st.container()
        with container:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Receipt", use_container_width=True)
            
            # Add button with same width as image
            st.markdown("""
                <style>
                    .stButton > button {
                        width: 100%;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            if st.button("Analyze Receipt 🔍", type="primary"):
                with st.spinner("Processing receipt..."):
                    try:
                        # Get image bytes
                        image_bytes = uploaded_file.getvalue()
                        
                        # Extract text using OCR for debugging
                        text_lines = extract_text_from_image(image_bytes)
                        
                        # Show extracted text in debug mode
                        if st.checkbox("Show extracted text (debug)"):
                            st.text("\n".join(text_lines))
                        
                        # Analyze receipt using Hugging Face model
                        response = analyze_receipt_with_model(image_bytes)
                        
                        # Store the response in session state
                        st.session_state['analysis_result'] = response
                        
                    except Exception as e:
                        st.error(f"Error processing receipt: {str(e)}")

# Main content area
if 'analysis_result' in st.session_state:
    # Create columns for results and copy button
    result_col, button_col = st.columns([4, 1])
    
    with result_col:
        st.markdown(st.session_state['analysis_result'])
    
    with button_col:
        # Add copy to clipboard button with JavaScript implementation
        st.markdown("""
            <style>
                .copy-button {
                    width: 100%;
                    padding: 0.5rem;
                    background-color: #0d6efd;
                    color: white;
                    border: none;
                    border-radius: 0.3rem;
                    cursor: pointer;
                }
                .copy-button:hover {
                    background-color: #0b5ed7;
                }
            </style>
        """, unsafe_allow_html=True)
        
        if st.button("📋 Copy to Clipboard", key="copy_btn"):
            # Store the markdown text in session state
            st.session_state['clipboard'] = st.session_state['analysis_result']
            
            # Create JavaScript to handle clipboard copy
            js_code = f"""
                <script>
                    // Function to copy text to clipboard
                    async function copyToClipboard() {{
                        const text = {repr(st.session_state['analysis_result'])};
                        try {{
                            await navigator.clipboard.writeText(text);
                            // Show success message
                            const btn = document.querySelector('.copy-button');
                            btn.textContent = '✓ Copied!';
                            setTimeout(() => {{
                                btn.textContent = '📋 Copy to Clipboard';
                            }}, 2000);
                        }} catch (err) {{
                            console.error('Failed to copy: ', err);
                        }}
                    }}
                    
                    // Call the function when button is clicked
                    document.querySelector('.copy-button').addEventListener('click', copyToClipboard);
                </script>
            """
            st.markdown(js_code, unsafe_allow_html=True)
            st.success("✓ Copied to clipboard!")
else:
    st.info("Upload a receipt image and click 'Analyze Receipt' to see the results here.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Hugging Face | [Report an Issue](https://github.com/yourusername/receipt-analyzer/issues)") 