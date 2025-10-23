"""
LLM-based intelligent field extraction for aviation documents
Uses OpenRouter API to extract section data based on aircraft/serial number search
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
    """Uses LLM via OpenRouter to extract data sections for specific identifiers"""
    
    # Standard field definitions
    STANDARD_FIELDS = {
        "engine_tsn": {
            "name": "Engine TSN (Time Since New)",
            "aliases": ["TSN", "Total Time Since New", "Total Hrs", "Engine TSN", "TAH"],
        },
        "engine_csn": {
            "name": "Engine CSN (Cycles Since New)",
            "aliases": ["CSN", "Total Cycles Since New", "Total Cyc", "Engine CSN", "TAC"],
        },
        "flight_hours_month": {
            "name": "Flight Hours This Month",
            "aliases": ["EFH operated", "Delta Hrs", "Hours this month", "Monthly hours"],
        },
        "flight_cycles_month": {
            "name": "Flight Cycles This Month",
            "aliases": ["EFC operated", "Delta Cyc", "Cycles this month", "Monthly cycles"],
        },
    }
    
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize LLM Field Extractor with OpenRouter"""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        
        if not self.api_key:
            raise ValueError("❌ OPENROUTER_API_KEY not set in .env file")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        logger.info(f"✅ LLM Field Extractor initialized (Model: {self.model})")
    
    def extract_column_data(self, ocr_results: Dict[str, Any], serial_number: str) -> Dict[str, Any]:
        """
        Extract data for a specific identifier (aircraft registration, serial number, etc.)
        Handles both columnar and row-based layouts
        
        Args:
            ocr_results: OCR results from Azure Computer Vision
            serial_number: Identifier to search for (e.g., "AKNT", "862909", "779682")
            
        Returns:
            Dictionary with extracted data and bounding boxes
        """
        logger.info(f"🔍 Searching for identifier: {serial_number}")
        
        # Prepare the prompt
        prompt = self._build_extraction_prompt(ocr_results, serial_number)
        
        try:
            # Call OpenRouter API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert aviation document analyzer. You extract all relevant data for a specific aircraft, engine, or component based on an identifier. You understand both columnar tables and row-based layouts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=4096
            )
            
            response_text = completion.choices[0].message.content
            logger.info(f"✅ LLM response received ({completion.usage.total_tokens} tokens)")
            
            # Parse response
            extracted_data = self._parse_llm_response(response_text)
            
            # Check if identifier was found
            if not extracted_data.get('identifier_found', False):
                return {
                    'found': False,
                    'message': f"Identifier '{serial_number}' not found in document",
                    'search_term': serial_number
                }
            
            # Enrich with bounding boxes
            enriched_data = self._enrich_extracted_data(extracted_data, ocr_results)
            
            logger.info(f"✅ Extracted {len(enriched_data.get('fields', {}))} fields")
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"❌ Error calling LLM: {str(e)}")
            raise
    
    def _build_extraction_prompt(self, ocr_results: Dict[str, Any], identifier: str) -> str:
        """Build the prompt for data extraction"""
        
        text_blocks = ocr_results.get('text_blocks', [])
        
        # Create simplified text blocks
        simplified_blocks = []
        for i, block in enumerate(text_blocks[:300]):
            simplified_blocks.append({
                "id": i,
                "text": block['text'],
                "x": round(block['bounding_box']['left']),
                "y": round(block['bounding_box']['top']),
                "width": round(block['bounding_box']['width']),
                "height": round(block['bounding_box']['height'])
            })
        
        prompt = f"""You are analyzing an aviation maintenance document. Find the identifier "{identifier}" and extract ALL related data.

**DOCUMENT LAYOUT TYPES:**
1. **Columnar Layout**: Data organized in vertical columns (e.g., engine status reports)
2. **Row-based Layout**: Data organized in horizontal rows/sections per aircraft (e.g., fleet reports)

**TASK:**
1. Find ALL occurrences of "{identifier}" in the text blocks
2. Determine the document layout type (columnar vs row-based)
3. For **COLUMNAR**: Extract all data in the same vertical column (±50px X tolerance)
4. For **ROW-BASED**: Extract all data in the same section/row (within ±150px Y from identifier)
5. Identify what each value represents

**OCR TEXT BLOCKS:**
```json
{json.dumps(simplified_blocks, indent=2)}
```

**AVIATION FIELD TYPES:**
- **aircraft_registration**: Aircraft registration (e.g., "AKNT", "VT-ANB")
- **msn**: Manufacturer Serial Number (e.g., "02607")
- **tah**: Total Aircraft Hours (format: "HHHH:MM" or decimal)
- **tac**: Total Aircraft Cycles (integer)
- **delta_hrs**: Hours this period (format: "HHH:MM" or decimal)
- **delta_cyc**: Cycles this period (integer)
- **c_check_expiry**: C-Check expiry date
- **d_check_expiry**: D-Check expiry date
- **engine_position**: Engine position (e.g., "1", "2", "#1", "Pos 1")
- **engine_pn**: Engine Part Number (e.g., "CFM56-5B6P")
- **engine_sn**: Engine Serial Number (e.g., "779682", "862909")
- **engine_tsn**: Engine Time Since New (total hours)
- **engine_csn**: Engine Cycles Since New (total cycles)
- **apu_sn**: APU Serial Number
- **apu_tsn**: APU Time Since New
- **apu_csn**: APU Cycles Since New
- **other**: Any other relevant data

**ANALYSIS STEPS:**
1. Find all blocks containing "{identifier}"
2. Analyze spatial relationships:
   - If data is vertically aligned (similar X): Extract COLUMN
   - If data is horizontally spread (similar Y): Extract ROW/SECTION
3. For row-based layouts, extract blocks within Y ±150px of the identifier
4. For columnar layouts, extract blocks within X ±50px of the identifier
5. Map each block to its field type

**RESPONSE FORMAT (JSON only):**
{{
  "identifier_found": true,
  "identifier": "{identifier}",
  "identifier_type": "aircraft_registration",  // or "engine_sn", "apu_sn", etc.
  "layout_type": "row_based",  // or "columnar"
  "extracted_fields": {{
    "aircraft_registration": {{
      "block_id": 26,
      "text": "AKNT",
      "field_type": "aircraft_registration"
    }},
    "msn": {{
      "block_id": 28,
      "text": "02607",
      "field_type": "msn"
    }},
    "tah": {{
      "block_id": 29,
      "text": "46865:23",
      "field_type": "tah"
    }},
    "tac": {{
      "block_id": 30,
      "text": "37968",
      "field_type": "tac"
    }},
    "delta_hrs": {{
      "block_id": 31,
      "text": "217:08",
      "field_type": "delta_hrs"
    }},
    "delta_cyc": {{
      "block_id": 32,
      "text": "164",
      "field_type": "delta_cyc"
    }},
    "engines": [
      {{
        "position": {{"block_id": 40, "text": "1", "field_type": "engine_position"}},
        "pn": {{"block_id": 41, "text": "CFM56-5B6P", "field_type": "engine_pn"}},
        "sn": {{"block_id": 42, "text": "779682", "field_type": "engine_sn"}},
        "tsn": {{"block_id": 43, "text": "55106:3", "field_type": "engine_tsn"}},
        "csn": {{"block_id": 44, "text": "40722", "field_type": "engine_csn"}},
        "delta_hrs": {{"block_id": 45, "text": "217:08", "field_type": "delta_hrs"}},
        "delta_cyc": {{"block_id": 46, "text": "164", "field_type": "delta_cyc"}}
      }}
    ]
  }},
  "bounding_box": {{
    "min_x": 100,
    "max_x": 1200,
    "min_y": 690,
    "max_y": 1100
  }}
}}

**IMPORTANT:**
- For aircraft registrations (like "AKNT"), extract the ENTIRE aircraft section including all engines
- For engine serial numbers, extract that specific engine's data
- Include check expiry dates if visible
- Group engine data together in an array if multiple engines present

If identifier not found: {{"identifier_found": false}}

Return ONLY the JSON object, nothing else.
"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data"""
        
        response_text = response_text.strip()
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1]
        if response_text.endswith('```'):
            response_text = response_text.rsplit('\n', 1)[0]
        
        response_text = response_text.strip()
        
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
    
    def _enrich_extracted_data(self, extracted_data: Dict[str, Any], 
                                ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """Add bounding box coordinates from OCR results to extracted fields"""
        
        text_blocks = ocr_results.get('text_blocks', [])
        extracted_fields = extracted_data.get('extracted_fields', {})
        
        enriched_fields = {}
        
        # Process flat fields
        for field_key, field_data in extracted_fields.items():
            if field_key == 'engines':
                # Handle engine array separately
                continue
                
            if isinstance(field_data, dict) and 'block_id' in field_data:
                block_id = field_data['block_id']
                if block_id < len(text_blocks):
                    ocr_block = text_blocks[block_id]
                    enriched_fields[field_key] = {
                        'text': field_data.get('text'),
                        'field_type': field_data.get('field_type'),
                        'bounding_box': ocr_block['bounding_box'],
                        'ocr_confidence': ocr_block.get('confidence', 'N/A')
                    }
        
        # Process engines array
        enriched_engines = []
        if 'engines' in extracted_fields:
            for engine in extracted_fields['engines']:
                enriched_engine = {}
                for key, field_data in engine.items():
                    if isinstance(field_data, dict) and 'block_id' in field_data:
                        block_id = field_data['block_id']
                        if block_id < len(text_blocks):
                            ocr_block = text_blocks[block_id]
                            enriched_engine[key] = {
                                'text': field_data.get('text'),
                                'field_type': field_data.get('field_type'),
                                'bounding_box': ocr_block['bounding_box'],
                                'ocr_confidence': ocr_block.get('confidence', 'N/A')
                            }
                enriched_engines.append(enriched_engine)
        
        if enriched_engines:
            enriched_fields['engines'] = enriched_engines
        
        # Calculate overall bounding box
        bbox_data = extracted_data.get('bounding_box', {})
        overall_bbox = {
            'left': bbox_data.get('min_x', 0),
            'top': bbox_data.get('min_y', 0),
            'width': bbox_data.get('max_x', 0) - bbox_data.get('min_x', 0),
            'height': bbox_data.get('max_y', 0) - bbox_data.get('min_y', 0)
        }
        
        return {
            'found': True,
            'identifier': extracted_data.get('identifier'),
            'identifier_type': extracted_data.get('identifier_type'),
            'layout_type': extracted_data.get('layout_type', 'unknown'),
            'fields': enriched_fields,
            'total_fields': len(enriched_fields),
            'bounding_box': overall_bbox,
            'document_type': self._detect_document_type(enriched_fields)
        }
    
    def _detect_document_type(self, fields: Dict[str, Any]) -> str:
        """Detect the type of aviation document based on extracted fields"""
        
        has_aircraft_data = 'aircraft_registration' in fields or 'msn' in fields
        has_tah_tac = 'tah' in fields or 'tac' in fields
        has_engines = 'engines' in fields
        has_engine_direct = any(k.startswith('engine_') for k in fields.keys())
        
        if has_aircraft_data and has_engines and has_tah_tac:
            return "Aircraft Lessor Report / Fleet Status"
        elif has_engine_direct:
            return "Engine Status Report"
        elif has_aircraft_data:
            return "Aircraft Status Report"
        else:
            return "Aviation Maintenance Document"


if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: python src/llm_field_extractor.py <ocr_json_file> <identifier>")
        print("\nExamples:")
        print('  python src/llm_field_extractor.py data/output/ocr_results/sample_page_1_ocr.json "862909"')
        print('  python src/llm_field_extractor.py data/output/ocr_results/sample3_page_1_ocr.json "AKNT"')
        print('  python src/llm_field_extractor.py data/output/ocr_results/sample3_page_1_ocr.json "779682"')
        sys.exit(1)
    
    json_file_path = Path(sys.argv[1])
    identifier = sys.argv[2]
    
    if not json_file_path.exists():
        print(f"❌ File not found: {json_file_path}")
        sys.exit(1)
    
    print(f"\n🧪 Testing Data Extraction")
    print(f"   File: {json_file_path.name}")
    print(f"   Identifier: {identifier}\n")
    
    # Load OCR results
    with open(json_file_path, 'r', encoding='utf-8') as f:
        ocr_results = json.load(f)
    
    # Extract data using LLM
    try:
        extractor = LLMFieldExtractor()
        result = extractor.extract_column_data(ocr_results, identifier)
        
        if not result['found']:
            print(f"❌ {result['message']}")
            sys.exit(1)
        
        # Print results
        print(f"✅ Found identifier: {result['identifier']}")
        print(f"   Type: {result['identifier_type']}")
        print(f"   Layout: {result['layout_type']}")
        print(f"   Document Type: {result['document_type']}")
        print(f"   Total fields: {result['total_fields']}\n")
        
        print(f"{'='*70}")
        print(f"Extracted Fields:")
        print(f"{'='*70}\n")
        
        for field_key, field_data in result['fields'].items():
            if field_key == 'engines':
                print(f"\n🔧 Engines:")
                for i, engine in enumerate(field_data, 1):
                    print(f"   Engine {i}:")
                    for key, data in engine.items():
                        print(f"      {key}: {data['text']}")
            else:
                print(f"  {field_key}: {field_data.get('text')}")
        
        # Bounding box
        bbox = result['bounding_box']
        print(f"\n{'='*70}")
        print(f"Overall Bounding Box:")
        print(f"{'='*70}")
        print(f"   Left: {bbox['left']:.0f}")
        print(f"   Top: {bbox['top']:.0f}")
        print(f"   Width: {bbox['width']:.0f}")
        print(f"   Height: {bbox['height']:.0f}\n")
        
        # Save results
        output_file = json_file_path.parent / f"{json_file_path.stem}_extracted_{identifier}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved to: {output_file.name}\n")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()