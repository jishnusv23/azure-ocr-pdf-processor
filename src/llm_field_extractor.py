"""
LLM-based intelligent field extraction for aviation documents
Uses OpenRouter API to extract column data based on serial number search
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMFieldExtractor:
    """Uses LLM via OpenRouter to extract column data for specific serial numbers"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM Field Extractor with OpenRouter
        
        Args:
            api_key: OpenRouter API key (uses OPENROUTER_API_KEY env var if not provided)
            model: Model to use (uses OPENROUTER_MODEL env var if not provided)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        
        if not self.api_key:
            raise ValueError("❌ OPENROUTER_API_KEY not set in .env file")
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        logger.info(f"✅ LLM Field Extractor initialized (Model: {self.model})")
    
    def extract_column_data(self, ocr_results: Dict[str, Any], serial_number: str) -> Dict[str, Any]:
        """
        Extract all data in the column containing the serial number
        
        Args:
            ocr_results: OCR results from Azure Computer Vision
            serial_number: Serial number to search for (e.g., "862909", "B3219")
            
        Returns:
            Dictionary with column data and bounding boxes
        """
        logger.info(f"🔍 Searching for serial number: {serial_number}")
        
        # Prepare the prompt
        prompt = self._build_column_extraction_prompt(ocr_results, serial_number)
        
        try:
            # Call OpenRouter API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert aviation document analyzer. You identify columns in tables and extract all data from a specific column containing a serial number."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=4096
            )
            
            # Get response
            response_text = completion.choices[0].message.content
            
            logger.info(f"✅ LLM response received ({completion.usage.total_tokens} tokens)")
            
            # Parse response
            extracted_data = self._parse_llm_response(response_text)
            
            # Enrich with bounding boxes
            enriched_data = self._enrich_column_data(extracted_data, ocr_results)
            
            logger.info(f"✅ Extracted {len(enriched_data.get('column_data', []))} items from column")
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"❌ Error calling LLM: {str(e)}")
            raise
    
    def _build_column_extraction_prompt(self, ocr_results: Dict[str, Any], serial_number: str) -> str:
        """Build the prompt for column data extraction"""
        
        # Get all text blocks
        text_blocks = ocr_results.get('text_blocks', [])
        
        # Create simplified text blocks for LLM
        simplified_blocks = []
        for i, block in enumerate(text_blocks[:300]):  # Increased limit for tables
            simplified_blocks.append({
                "id": i,
                "text": block['text'],
                "x": round(block['bounding_box']['left']),
                "y": round(block['bounding_box']['top']),
                "width": round(block['bounding_box']['width']),
                "height": round(block['bounding_box']['height'])
            })
        
        # Build the prompt
        prompt = f"""You are analyzing an aviation maintenance document with tables. Find the serial number "{serial_number}" and extract ALL data from that column.

**TASK:**
1. Find text block(s) containing "{serial_number}"
2. Identify which COLUMN this serial number belongs to (based on X position)
3. Extract ALL text blocks in that same column (similar X position, ±50 pixels)
4. Identify what each value represents (TSN, CSN, Hours, Cycles, etc.)

**OCR TEXT BLOCKS:**
```json
{json.dumps(simplified_blocks, indent=2)}
```

**AVIATION FIELD TYPES:**
- **Header**: Column header (e.g., "POSITION NO.1", "Engine #1", "APU", "Nose Landing Gear")
- **Serial Number**: Engine/APU/component serial number (e.g., "862909", "B3219", "P-11217")
- **TSN (Time Since New)**: Total hours in format "HHHH:MM" (e.g., "48628:14", "16300")
- **CSN (Cycles Since New)**: Total cycles as integer (e.g., "30220", "8200")
- **Location**: Installation location (e.g., "A-7575")
- **Monthly Hours**: Hours this period (e.g., "197.25", "226:17")
- **Monthly Cycles**: Cycles this period (e.g., "230", "194")
- **Part Number**: P/N (e.g., "P-11217")

**INSTRUCTIONS:**
- Group blocks by X position to identify columns (±50 pixels tolerance)
- For the column containing "{serial_number}", extract ALL blocks
- Sort by Y position (top to bottom)
- Identify the field type for each block
- Return block IDs and field types

**RESPONSE FORMAT (JSON only):**
{{
  "serial_number_found": true,
  "serial_number": "{serial_number}",
  "serial_number_block_id": 15,
  "column_x_position": 1715,
  "column_data": [
    {{
      "block_id": 12,
      "text": "POSITION NO.1",
      "field_type": "header",
      "y_position": 1850
    }},
    {{
      "block_id": 13,
      "text": "862909",
      "field_type": "serial_number",
      "y_position": 2002
    }},
    {{
      "block_id": 16,
      "text": "16300",
      "field_type": "tsn",
      "y_position": 2443
    }},
    {{
      "block_id": 17,
      "text": "8200",
      "field_type": "csn",
      "y_position": 2590
    }},
    {{
      "block_id": 18,
      "text": "197.25",
      "field_type": "monthly_hours",
      "y_position": 2737
    }},
    {{
      "block_id": 19,
      "text": "230",
      "field_type": "monthly_cycles",
      "y_position": 2884
    }}
  ],
  "column_bounding_box": {{
    "min_x": 1650,
    "max_x": 1850,
    "min_y": 1850,
    "max_y": 2900
  }}
}}

If serial number not found, return: {{"serial_number_found": false}}

Return ONLY the JSON object, nothing else.
"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data"""
        
        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1]
        if response_text.endswith('```'):
            response_text = response_text.rsplit('\n', 1)[0]
        
        response_text = response_text.strip()
        
        # Find JSON in response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON found in LLM response")
        
        json_str = response_text[start:end]
        
        try:
            extracted_data = json.loads(json_str)
            return extracted_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {json_str[:500]}...")
            raise
    
    def _enrich_column_data(self, extracted_data: Dict[str, Any], 
                           ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add full bounding box information from OCR results
        
        Args:
            extracted_data: Data extracted by LLM
            ocr_results: Original OCR results with bounding boxes
            
        Returns:
            Enriched data with complete bounding boxes
        """
        text_blocks = ocr_results.get('text_blocks', [])
        
        if not extracted_data.get('serial_number_found'):
            return {
                "found": False,
                "serial_number": extracted_data.get('serial_number'),
                "message": "Serial number not found in document"
            }
        
        enriched_column_data = []
        
        for item in extracted_data.get('column_data', []):
            block_id = item.get('block_id')
            
            if block_id is not None and block_id < len(text_blocks):
                block = text_blocks[block_id]
                
                enriched_column_data.append({
                    "text": item['text'],
                    "field_type": item['field_type'],
                    "bounding_box": block['bounding_box'],
                    "ocr_confidence": block.get('confidence'),
                    "y_position": item['y_position']
                })
        
        # Calculate overall column bounding box
        if enriched_column_data:
            min_left = min(item['bounding_box']['left'] for item in enriched_column_data)
            max_right = max(item['bounding_box']['right'] for item in enriched_column_data)
            min_top = min(item['bounding_box']['top'] for item in enriched_column_data)
            max_bottom = max(item['bounding_box']['bottom'] for item in enriched_column_data)
            
            column_bbox = {
                "left": min_left,
                "top": min_top,
                "right": max_right,
                "bottom": max_bottom,
                "width": max_right - min_left,
                "height": max_bottom - min_top
            }
        else:
            column_bbox = extracted_data.get('column_bounding_box', {})
        
        return {
            "found": True,
            "serial_number": extracted_data.get('serial_number'),
            "column_data": enriched_column_data,
            "column_bounding_box": column_bbox,
            "total_items": len(enriched_column_data)
        }


if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: python src/llm_field_extractor.py <ocr_json_file> <serial_number>")
        print("\nExample:")
        print('  python src/llm_field_extractor.py data/output/ocr_results/sample_page_1_ocr.json "862909"')
        sys.exit(1)
    
    # Get file path and serial number from command line
    json_file_path = Path(sys.argv[1])
    serial_number = sys.argv[2]
    
    if not json_file_path.exists():
        print(f"❌ File not found: {json_file_path}")
        sys.exit(1)
    
    print(f"\n🧪 Testing Column Extraction")
    print(f"   File: {json_file_path.name}")
    print(f"   Serial Number: {serial_number}\n")
    
    # Load OCR results
    with open(json_file_path, 'r', encoding='utf-8') as f:
        ocr_results = json.load(f)
    
    # Extract column data using LLM
    try:
        extractor = LLMFieldExtractor()
        result = extractor.extract_column_data(ocr_results, serial_number)
        
        if not result['found']:
            print(f"❌ {result['message']}")
            sys.exit(1)
        
        # Print results
        print(f"✅ Found serial number: {result['serial_number']}")
        print(f"   Total items in column: {result['total_items']}\n")
        
        print(f"{'='*70}")
        print(f"Column Data (sorted top to bottom):")
        print(f"{'='*70}\n")
        
        for i, item in enumerate(result['column_data'], 1):
            bbox = item['bounding_box']
            print(f"{i:2d}. {item['field_type']:<20}: {item['text']}")
            print(f"    Position: ({bbox['left']:.0f}, {bbox['top']:.0f})")
            print(f"    Size: {bbox['width']:.0f} x {bbox['height']:.0f}")
            print(f"    Confidence: {item.get('ocr_confidence', 'N/A')}")
            print()
        
        # Column bounding box for highlighting
        col_bbox = result['column_bounding_box']
        print(f"{'='*70}")
        print(f"Column Bounding Box (for highlighting entire column):")
        print(f"{'='*70}")
        print(f"   Left: {col_bbox['left']:.0f}")
        print(f"   Top: {col_bbox['top']:.0f}")
        print(f"   Width: {col_bbox['width']:.0f}")
        print(f"   Height: {col_bbox['height']:.0f}\n")
        
        # Save enriched data
        output_file = json_file_path.parent / f"{json_file_path.stem}_column_{serial_number}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved to: {output_file.name}\n")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()