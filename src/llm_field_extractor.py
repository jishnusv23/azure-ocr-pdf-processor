"""
LLM-based intelligent field extraction for aviation documents
Uses OpenRouter API with Instructor for structured output
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
import logging
from openai import OpenAI
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field
from src.pdf_highlighter import highlight_extraction_in_pdf
from src.type.index import ExtractionResult


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





class LLMFieldExtractor:
    """Uses LLM via OpenRouter with Instructor for structured extraction"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize LLM Field Extractor with Instructor"""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        
        if not self.api_key:
            raise ValueError("❌ OPENROUTER_API_KEY not set in .env file")
        
        # Initialize OpenAI client
        base_client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Patch with Instructor
        self.client = instructor.from_openai(base_client)
        
        logger.info(f"✅ LLM Field Extractor initialized with Instructor (Model: {self.model})")
    
    def extract_column_data(self, ocr_results: Dict[str, Any], serial_number: str) -> Dict[str, Any]:
        """
        Extract data for a specific identifier using Instructor for structured output
        
        Args:
            ocr_results: OCR results from Azure Computer Vision
            serial_number: Identifier to search for
            
        Returns:
            Dictionary with extracted data and bounding boxes
        """
        logger.info(f"🔍 Searching for identifier: {serial_number}")
        
        # Prepare text blocks
        text_blocks = ocr_results.get('text_blocks', [])
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
        
        # Build prompt
        prompt = f"""You are analyzing an aviation maintenance document. Find the identifier "{serial_number}" and extract ALL related data.

**DOCUMENT LAYOUT TYPES:**
1. **Columnar Layout**: Data organized in vertical columns
2. **Row-based Layout**: Data organized in horizontal rows/sections

**OCR TEXT BLOCKS:**
{json.dumps(simplified_blocks, indent=2)}

**TASK:**
1. Find all occurrences of "{serial_number}"
2. Determine layout type (columnar vs row-based)
3. For **COLUMNAR**: Extract data in same vertical column (±50px X)
4. For **ROW-BASED**: Extract data in same section (±150px Y)
5. Map each value to its field type

**FIELD TYPES:**
- apu_sn, apu_tsn, apu_csn (APU data)
- engine_sn, engine_tsn, engine_csn (Engine data)
- aircraft_registration, msn (Aircraft data)
- delta_hrs, delta_cyc (Period data)

Extract ALL fields in the same column/section as the identifier.
"""
        
        try:
            # Call Instructor-patched client
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=ExtractionResult,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert aviation document analyzer. Extract structured data for aircraft, engines, and APUs."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=4096
            )
            
            logger.info(f"✅ LLM response received (structured with Instructor)")
            
            # Check if identifier was found
            if not result.identifier_found:
                return {
                    'found': False,
                    'message': f"Identifier '{serial_number}' not found in document",
                    'search_term': serial_number
                }
            
            # Convert to dictionary and enrich with bounding boxes
            extracted_dict = result.model_dump()
            enriched_data = self._enrich_extracted_data(extracted_dict, ocr_results)
            
            logger.info(f"✅ Extracted {len(enriched_data.get('fields', {}))} fields")
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"❌ Error calling LLM: {str(e)}")
            raise
    
    def _enrich_extracted_data(self, extracted_data: Dict[str, Any], 
                                ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """Add bounding box coordinates from OCR results to extracted fields"""
        
        text_blocks = ocr_results.get('text_blocks', [])
        extracted_fields = extracted_data.get('extracted_fields', {})
        
        enriched_fields = {}
        
        # Process flat fields
        for field_key, field_data in extracted_fields.items():
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
        engines = extracted_data.get('engines', [])
        
        if engines:
            for engine in engines:
                enriched_engine = {}
                engine_dict = engine if isinstance(engine, dict) else engine.model_dump()
                
                for key, field_data in engine_dict.items():
                    if field_data and isinstance(field_data, dict) and 'block_id' in field_data:
                        block_id = field_data['block_id']
                        if block_id < len(text_blocks):
                            ocr_block = text_blocks[block_id]
                            enriched_engine[key] = {
                                'text': field_data.get('text'),
                                'field_type': field_data.get('field_type'),
                                'bounding_box': ocr_block['bounding_box'],
                                'ocr_confidence': ocr_block.get('confidence', 'N/A')
                            }
                
                if enriched_engine:
                    enriched_engines.append(enriched_engine)
        
        if enriched_engines:
            enriched_fields['engines'] = enriched_engines
        
        # Calculate overall bounding box
        bbox_data = extracted_data.get('bounding_box', {})
        if bbox_data:
            bbox_dict = bbox_data if isinstance(bbox_data, dict) else bbox_data.model_dump()
            overall_bbox = {
                'left': bbox_dict.get('min_x', 0),
                'top': bbox_dict.get('min_y', 0),
                'width': bbox_dict.get('max_x', 0) - bbox_dict.get('min_x', 0),
                'height': bbox_dict.get('max_y', 0) - bbox_dict.get('min_y', 0)
            }
        else:
            overall_bbox = {'left': 0, 'top': 0, 'width': 0, 'height': 0}
        
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
        has_apu = any(k.startswith('apu_') for k in fields.keys())
        
        if has_aircraft_data and has_engines and has_tah_tac:
            return "Aircraft Lessor Report / Fleet Status"
        elif has_apu:
            return "APU Status Report"
        elif has_engines:
            return "Engine Status Report"
        elif has_aircraft_data:
            return "Aircraft Status Report"
        else:
            return "Aviation Maintenance Document"


if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: python src/llm_field_extractor.py <ocr_json_file> <identifier> [pdf_file]")
        print("\nExamples:")
        print('  python src/llm_field_extractor.py sample_page_1_ocr.json "P-11217"')
        print('  python src/llm_field_extractor.py sample_page_1_ocr.json "P-11217" input.pdf')
        sys.exit(1)
    
    json_file_path = Path(sys.argv[1])
    identifier = sys.argv[2]
    pdf_file_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not json_file_path.exists():
        print(f"❌ File not found: {json_file_path}")
        sys.exit(1)
    
    print(f"\n🧪 Testing Data Extraction with Instructor")
    print(f"   OCR File: {json_file_path.name}")
    print(f"   Identifier: {identifier}")
    if pdf_file_path:
        print(f"   PDF File: {pdf_file_path.name}")
    print()
    
    # Load OCR results
    with open(json_file_path, 'r', encoding='utf-8') as f:
        ocr_results = json.load(f)
    
    # Extract data using LLM with Instructor
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
        
        # Save JSON results
        output_file = json_file_path.parent / f"{json_file_path.stem}_extracted_{identifier}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 JSON saved to: {output_file.name}\n")
        
        # Highlight in PDF if provided
        if pdf_file_path and pdf_file_path.exists():
            print(f"{'='*70}")
            print(f"🎨 Highlighting in PDF...")
            print(f"{'='*70}\n")
            
            highlighted_pdf = highlight_extraction_in_pdf(
                pdf_path=str(pdf_file_path),
                extraction_result=result,
                method="rectangle",
                ocr_results=ocr_results
            )
            
            print(f"\n✅ Highlighted PDF created: {Path(highlighted_pdf).name}")
            print(f"   Location: {highlighted_pdf}\n")
        elif pdf_file_path:
            print(f"\n⚠️  PDF file not found: {pdf_file_path}")
            print(f"   Skipping PDF highlighting...\n")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()