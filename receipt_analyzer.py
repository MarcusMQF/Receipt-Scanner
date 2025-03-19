from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from typing import Dict, List, Optional
from PIL import Image

class ReceiptAnalyzer:
    def __init__(self):
        # Initialize with openai-community/gpt2
        self.model_name = "openai-community/gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def analyze_receipt(self, text: str) -> Dict:
        """
        Analyze receipt text and extract relevant information
        
        Args:
            text (str): The extracted text from the receipt
            
        Returns:
            Dict: Dictionary containing extracted information
        """
        # Prepare the prompt
        prompt = f"""Extract the following information from this receipt:
1. Restaurant name and details
2. Date and time
3. Individual items with prices
4. Tax amount
5. Total amount

Receipt text:
{text}

Analysis:
Restaurant Details:"""

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=1024,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                attention_mask=inputs.attention_mask
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Process the response into structured data
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Dict:
        """
        Parse the model's response into structured data
        
        Args:
            response (str): Raw response from the model
            
        Returns:
            Dict: Structured data containing receipt information
        """
        try:
            # Extract sections from the response
            lines = response.split('\n')
            result = {
                'restaurant_details': {},
                'items': [],
                'tax': 0.0,
                'total': 0.0,
                'date_time': ''
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if 'restaurant' in line.lower():
                    current_section = 'restaurant'
                    continue
                elif 'items' in line.lower() or 'order' in line.lower():
                    current_section = 'items'
                    continue
                elif 'tax' in line.lower():
                    try:
                        result['tax'] = float(''.join(filter(str.isdigit, line.split(':')[1].strip())))
                    except:
                        pass
                elif 'total' in line.lower():
                    try:
                        result['total'] = float(''.join(filter(str.isdigit, line.split(':')[1].strip())))
                    except:
                        pass
                elif 'date' in line.lower():
                    result['date_time'] = line.split(':')[1].strip()
                    
                if current_section == 'restaurant':
                    if ':' in line:
                        key, value = line.split(':', 1)
                        result['restaurant_details'][key.strip()] = value.strip()
                elif current_section == 'items':
                    if '-' in line or ':' in line or '$' in line:
                        parts = line.replace('$', '').split('-')
                        if len(parts) == 2:
                            item_name = parts[0].strip()
                            try:
                                price = float(''.join(filter(str.isdigit, parts[1].strip())))
                                result['items'].append({
                                    'item': item_name,
                                    'price': price
                                })
                            except:
                                pass
                                
            return result
        except Exception as e:
            return {
                'error': f"Failed to parse response: {str(e)}",
                'raw_response': response
            }
    
    def format_as_markdown(self, analysis: Dict) -> str:
        """
        Format the analysis results as markdown text
        
        Args:
            analysis (Dict): Analysis results
            
        Returns:
            str: Formatted markdown text
        """
        markdown = []
        
        # Restaurant Details
        markdown.append("## Restaurant Details")
        for key, value in analysis.get('restaurant_details', {}).items():
            markdown.append(f"**{key}:** {value}")
        
        # Date and Time
        if analysis.get('date_time'):
            markdown.append(f"\n**Date/Time:** {analysis['date_time']}")
        
        # Items
        markdown.append("\n## Items")
        markdown.append("| Item | Price |")
        markdown.append("|------|-------|")
        for item in analysis.get('items', []):
            markdown.append(f"| {item['item']} | ${item['price']:.2f} |")
        
        # Tax and Total
        markdown.append(f"\n**Tax:** ${analysis.get('tax', 0):.2f}")
        markdown.append(f"**Total:** ${analysis.get('total', 0):.2f}")
        
        return "\n".join(markdown)
    
    def format_as_table(self, analysis: Dict) -> pd.DataFrame:
        """
        Format the items as a pandas DataFrame
        
        Args:
            analysis (Dict): Analysis results
            
        Returns:
            pd.DataFrame: Formatted table of items
        """
        items_df = pd.DataFrame(analysis.get('items', []))
        if not items_df.empty:
            items_df['price'] = items_df['price'].apply(lambda x: f"${x:.2f}")
        return items_df 