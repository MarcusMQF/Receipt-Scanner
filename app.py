import streamlit as st
from huggingface_hub import InferenceClient
import base64
from PIL import Image
import io
import os

# Initialize Hugging Face client
def init_hf_client():
    if 'hf_client' not in st.session_state:
        api_key = st.secrets.get("HUGGING_FACE_API_KEY") or os.getenv("HUGGING_FACE_API_KEY")
        if not api_key:
            st.error("Please set the HUGGING_FACE_API_KEY in your environment variables or Streamlit secrets.")
            st.stop()
        st.session_state['hf_client'] = InferenceClient(
            provider="hyperbolic",
            api_key=api_key
        )
    return st.session_state['hf_client']

# Page configuration
st.set_page_config(
    page_title="Receipt Analyzer",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description in main area
st.markdown("""
    # 🧾 Receipt Analyzer
    Extract and analyze receipt information using AI! Results are formatted in markdown for easy copying.
""")

# Add clear button to top right
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear 🗑️"):
        if 'analysis_result' in st.session_state:
            del st.session_state['analysis_result']
        st.rerun()

st.markdown("---")

# Move upload controls to sidebar
with st.sidebar:
    st.header("Upload Receipt")
    uploaded_file = st.file_uploader("Choose a receipt image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Receipt")
        
        if st.button("Analyze Receipt 🔍", type="primary"):
            with st.spinner("Processing receipt..."):
                try:
                    # Initialize HF client
                    client = init_hf_client()
                    
                    # Convert image to base64
                    image_bytes = uploaded_file.getvalue()
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Prepare the messages for the model
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Analyze this receipt image and format the output EXACTLY as follows with consistent spacing:

### Restaurant Details

   - Name
   - Address
   - Phone number 
   - Email

&nbsp;

### Bill Details

   - Tax ID/GST number
   - Date
   - Time
   - Table number
   - Server
   - Other relevant details

&nbsp;                     
                                    
### Order Details

| Number | Item | Quantity | Price Before Tax | Price After Tax | Total |
|--------|------|----------|------------------|-----------------|-------|
| 1 | [full item name, combining multi-line items] | [qty] | [amount] | [amount] | RM [amount] |
[continue for all items...]


&nbsp;

### Tax 

| Type | Percentage | Amount |
|------|------------|--------|
| SST | [percentage]% | RM [amount] |


&nbsp;

### Total

- Subtotal: RM [amount]
- Tax: RM [amount]
- Final Total: RM [amount]

Note: 
1. Each section MUST be separated by TWO blank lines (use '&nbsp;' between sections)
2. All currency values must have 'RM' prefix
3. All numbers must have 2 decimal places
4. Keep exact item names but COMBINE multi-line items into single items
5. Ensure proper table column alignment
6. Add percentage symbol (%) after tax percentage
7. Only display sections that are present in the receipt"""
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
                    
                    # Get completion from model
                    completion = client.chat.completions.create(
                        model="Qwen/Qwen2.5-VL-7B-Instruct",
                        messages=messages,
                        max_tokens=1000
                    )
                    
                    # Get the response text and format it
                    if completion and completion.choices:
                        response_text = completion.choices[0].message.content
                        
                        # Add copy button section at the bottom
                        formatted_text = response_text + "\n\n"
                        
                        st.session_state['analysis_result'] = formatted_text
                    else:
                        st.error("No response received from the model")
                    
                except Exception as e:
                    st.error(f"Error processing receipt: {str(e)}")

# Main content area for results
if 'analysis_result' in st.session_state:
    # Create columns for the results and copy button
    result_col, button_col = st.columns([4, 1])
    
    with result_col:
        # Display results
        st.markdown(st.session_state['analysis_result'])
    
    with button_col:
        # Add copy to clipboard button
        if st.button("📋 Copy to Clipboard"):
            st.write("Results copied to clipboard!")
            st.session_state['clipboard'] = st.session_state['analysis_result']
else:
    st.info("Upload a receipt image and click 'Analyze Receipt' to see the results here.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Qwen Vision-Language Model | [Report an Issue](https://github.com/yourusername/receipt-analyzer/issues)") 