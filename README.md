# Receipt Scanner

A Streamlit-based application that analyzes receipt images using AI to extract and format receipt information.

## Features

- Upload and analyze receipt images
- Extract restaurant details, bill details, order items, tax information, and totals
- Format results in clean, readable markdown
- Copy results to clipboard
- Clean and intuitive user interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MarcusMQF/Receipt-Scanner.git
cd Receipt-Scanner
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Hugging Face API key to the `.env` file
   - Or set the environment variable directly:
     ```bash
     # On Windows (PowerShell)
     $env:HUGGING_FACE_API_KEY="your_api_key_here"
     
     # On Linux/Mac
     export HUGGING_FACE_API_KEY="your_api_key_here"
     ```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the displayed URL (typically http://localhost:8501)

3. Upload a receipt image and click "Analyze Receipt"

## Requirements

- Python 3.8 or higher
- Dependencies listed in requirements.txt
- Hugging Face API key

## License

MIT License 