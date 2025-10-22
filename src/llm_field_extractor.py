"""
LLM-based intelligent field extraction for aviation documents
Uses OpenRouter API to access Claude and other models
"""
import json
import os
from typing import Dict, Any, List, Optional
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMFieldExtractor:
    """Uses LLM via OpenRouter to intelligently extract aviation fields from OCR data"""
    
    # Standard field definitions
    STANDARD_FIELDS = {
        "engine_tsn": {
            "name": "Engine TSN (Time Since New)",
            "aliases": ["TSN", "Total Time Since New", "Total Hrs", "Engine TSN", "TAH", "Total Time"],
            "description": "Total operating hours of the engine since new"
        },
        "engine_csn": {
            "name": "Engine CSN (Cycles Since New)",
            "aliases": ["CSN", "Total Cycles Since New", "Total Cyc", "Engine CSN", "TAC", "Total Cycles"],
            "description": "Total operating cycles of the engine since new"
        },
        "flight_hours_month": {
            "name": "Flight Hours This Month",
            "aliases": ["EFH operated", "Delta Hrs", "Hours this month", "Monthly hours", "EFH this month"],
            "description": "Flight hours operated during the reporting period"
        },
        "flight_cycles_month": {
            "name": "Flight Cycles This Month",
            "aliases": ["EFC operated", "Delta Cyc", "Cycles this month", "Monthly cycles", "EFC this month"],
            "description": "Flight cycles operated during the reporting period"
        },
        "engine_serial_number": {
            "name": "Engine Serial Number",
            "aliases": ["ESN", "Engine S/N", "Serial Number", "S/N", "SN"],
            "description": "Unique serial number of the engine"
        },
        "aircraft_registration": {
            "name": "Aircraft Registration",
            "aliases": ["Registration", "Aircraft Reg", "Reg", "A/C Registration"],
            "description": "Aircraft registration code"
        },
        "engine_position": {
            "name": "Engine Position",
            "aliases": ["Position", "Pos", "Engine Pos", "Installation position"],
            "description": "Engine installation position (e.g., #1, #2)"
        },
        "operator": {
            "name": "Operator",
            "aliases": ["Operator", "Airline", "Company"],
            "description": "Aircraft operator name"
        },
        "aircraft_msn": {
            "name": "Aircraft MSN",
            "aliases": ["MSN", "Aircraft serial number", "Serial number"],
            "description": "Manufacturer Serial Number"
        }
    }
    
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
    
    def extract_fields(self, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to extract and standardize aviation fields from OCR results
        
        Args:
            ocr_results: OCR results from Azure Computer Vision
            
        Returns:
            Dictionary with standardized fields and their bounding boxes
        """
        logger.info("🧠 Using LLM to extract fields...")
        
        # Prepare the prompt
        prompt = self._build_extraction_prompt(ocr_results)
        
        try:
            # Call OpenRouter API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert aviation document analyzer specializing in maintenance reports, utilization reports, and lessor reports. You extract structured data from OCR results."
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
            
            logger.info(f"✅ LLM extracted {len(extracted_data.get('fields', {}))} fields")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"❌ Error calling LLM: {str(e)}")
            raise
    
    def _build_extraction_prompt(self, ocr_results: Dict[str, Any]) -> str:
        """Build the prompt for LLM field extraction"""
        
        # Get all text blocks
        text_blocks = ocr_results.get('text_blocks', [])
        
        # Create simplified text blocks for LLM (limit to first 200 blocks to save tokens)
        simplified_blocks = []
        for i, block in enumerate(text_blocks[:200]):
            simplified_blocks.append({
                "id": i,
                "text": block['text'],
                "x": round(block['bounding_box']['left']),
                "y": round(block['bounding_box']['top']),
                "width": round(block['bounding_box']['width']),
                "height": round(block['bounding_box']['height'])
            })
        
        # Build the prompt
        prompt = f"""You are analyzing an aviation maintenance document. Extract the following fields:

**FIELDS TO EXTRACT:**
1. Engine TSN (Time Since New) - Total hours, may appear as "TSN", "Total Hrs", "TAH"
2. Engine CSN (Cycles Since New) - Total cycles, may appear as "CSN", "Total Cyc", "TAC"
3. Flight Hours This Month - Monthly hours, may appear as "EFH operated", "Delta Hrs"
4. Flight Cycles This Month - Monthly cycles, may appear as "EFC operated", "Delta Cyc"
5. Engine Serial Number (ESN) - May appear as "ESN", "S/N", "Serial Number"
6. Aircraft Registration - May appear as "Registration", "A/C Reg"
7. Engine Position - May appear as "Position", "Pos", "#1", "#2"
8. Operator - Airline/company name
9. Aircraft MSN - Manufacturer serial number
**OCR TEXT BLOCKS:**
```json
{json.dumps(simplified_blocks, indent=2)}
```

**INSTRUCTIONS:**
- For each field, find the LABEL block and the VALUE block (usually right or below the label)
- Values are typically within 100 pixels of labels
- Aviation time format: "HH:MM" or "HHHH:MM" (e.g., "48628:14")
- Return ONLY fields you found with high confidence

**RESPONSE FORMAT (JSON only, no markdown):**
{{
  "fields": {{
    "engine_tsn": {{
      "label_block_id": 12,
      "value_block_id": 13,
      "value": "48628:14",
      "confidence": "high"
    }},
    "engine_csn": {{
      "label_block_id": 14,
      "value_block_id": 15,
      "value": "30220",
      "confidence": "high"
    }}
  }}
}}

Return ONLY the JSON object, nothing else.
"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data"""
        
        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith('```'):
            # Remove ```json or ``` from start
            response_text = response_text.split('\n', 1)[1]
        if response_text.endswith('```'):
            # Remove ``` from end
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
            logger.error(f"Failed to parse JSON: {json_str[:200]}...")
            raise
    
    def enrich_with_bounding_boxes(self, extracted_data: Dict[str, Any], 
                                   ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add bounding box coordinates from OCR results to extracted fields
        
        Args:
            extracted_data: Data extracted by LLM
            ocr_results: Original OCR results with bounding boxes
            
        Returns:
            Enriched data with bounding boxes for highlighting
        """
        text_blocks = ocr_results.get('text_blocks', [])
        
        enriched_fields = {}
        
        for field_key, field_data in extracted_data.get('fields', {}).items():
            value_block_id = field_data.get('value_block_id')
            label_block_id = field_data.get('label_block_id')
            
            if value_block_id is not None and value_block_id < len(text_blocks):
                value_block = text_blocks[value_block_id]
                
                enriched_fields[field_key] = {
                    "standard_name": self.STANDARD_FIELDS.get(field_key, {}).get('name', field_key),
                    "value": field_data.get('value'),
                    "confidence": field_data.get('confidence'),
                    "bounding_box": value_block['bounding_box'],
                    "ocr_confidence": value_block.get('confidence'),
                    "label_block_id": label_block_id,
                    "value_block_id": value_block_id
                }
                
                # Add label bounding box if available
                if label_block_id is not None and label_block_id < len(text_blocks):
                    label_block = text_blocks[label_block_id]
                    enriched_fields[field_key]['label_bounding_box'] = label_block['bounding_box']
                    enriched_fields[field_key]['label_text'] = label_block['text']
        
        return {
            "extracted_fields": enriched_fields,
            "total_fields_found": len(enriched_fields),
            "document_type": self._detect_document_type(enriched_fields)
        }
    
    def _detect_document_type(self, fields: Dict[str, Any]) -> str:
        """Detect the type of aviation document based on extracted fields"""
        
        field_keys = set(fields.keys())
        
        if "flight_hours_month" in field_keys and "flight_cycles_month" in field_keys:
            return "Utilization Report"
        elif "engine_position" in field_keys:
            return "Aircraft Lessor Report"
        else:
            return "Aviation Maintenance Document"


if __name__ == "__main__":
    # Test the LLM extractor
    from config.config import OCR_RESULTS_DIR
    
    # Find a test OCR file
    ocr_files = list(OCR_RESULTS_DIR.glob("*.json"))
    
    if not ocr_files:
        print("❌ No OCR files found for testing")
    else:
        test_file = ocr_files[0]
        print(f"\n🧪 Testing with: {test_file.name}\n")
        
        # Load OCR results
        with open(test_file, 'r', encoding='utf-8') as f:
            ocr_results = json.load(f)
        
        # Extract fields using LLM
        try:
            extractor = LLMFieldExtractor()
            extracted = extractor.extract_fields(ocr_results)
            
            # Enrich with bounding boxes
            enriched = extractor.enrich_with_bounding_boxes(extracted, ocr_results)
            
            # Print results
            print(f"📊 Extraction Results:")
            print(f"   Document Type: {enriched['document_type']}")
            print(f"   Fields Found: {enriched['total_fields_found']}\n")
            
            for field_key, field_data in enriched['extracted_fields'].items():
                print(f"✅ {field_data['standard_name']}")
                print(f"   Value: {field_data['value']}")
                print(f"   Confidence: {field_data['confidence']}")
                print(f"   Bounding Box: {field_data['bounding_box']}")
                if 'label_text' in field_data:
                    print(f"   Label: {field_data['label_text']}")
                print()
            
            # Save enriched data
            output_file = OCR_RESULTS_DIR / f"{test_file.stem}_llm_extracted.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enriched, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")