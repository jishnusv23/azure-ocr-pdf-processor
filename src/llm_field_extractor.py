"""
Enhanced LLM Field Extractor - COMPLETE MULTI-PAGE SUPPORT with ExtractedComponentData
Location: src/llm_field_extractor.py
"""
import json
import os
from typing import Dict, Any, List, Optional
import logging
from openai import OpenAI
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field

from src.model.extractor_model import (
    BoundingBox, 
    ExtractedComponentData,
    ComponentData,
    DocumentType
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# LLM FIELD EXTRACTOR
# ============================================================================

class LLMFieldExtractor:
    """LLM Field Extractor - Multi-Page Support using ExtractedComponentData structure"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/o3")
        
        if not self.api_key:
            raise ValueError("❌ OPENROUTER_API_KEY not set in .env file")
        
        base_client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.client = instructor.from_openai(base_client)
        logger.info(f"✅ LLM Field Extractor initialized (Model: {self.model})")
    
    def extract_all_data_multipage(self, all_ocr_data: List[Dict[str, Any]], 
                                   pdf_filename: str) -> Dict[str, Any]:
        """
        MULTI-PAGE EXTRACTION using ExtractedComponentData structure
        Returns component data organized by Airframe, Engine1, Engine2, APU, Landing Gears
        
        Args:
            all_ocr_data: List of OCR data dicts, one per page
            pdf_filename: Name of PDF file
            
        Returns:
            Dict with structure:
            {
                'Airframe': {'TSN': 12345, 'TSN_bbox': {...}, 'CSN': 6789, ...},
                'Engine1': {'SerialNumber': '862909', 'TSN': 16300, ...},
                'Engine2': {...},
                'APU': {...},
                'LandingGearLeft': {...},
                'LandingGearRight': {...},
                'LandingGearNose': {...}
            }
        """
        logger.info(f"🔍 Multi-page extraction: Processing {len(all_ocr_data)} pages together...")
        
        # Combine all OCR blocks from all pages
        combined_blocks = []
        page_to_blocks = {}  # Map page number to its block indices
        
        block_id = 0
        for ocr_data in all_ocr_data:
            page_num = ocr_data['page_number']
            page_blocks_start = block_id
            
            for block in ocr_data['text_blocks']:
                combined_blocks.append({
                    "id": block_id,
                    "page": page_num,
                    "text": block['text'],
                    "bbox": {
                        "left": round(block['bounding_box']['left']),
                        "top": round(block['bounding_box']['top']),
                        "width": round(block['bounding_box']['width']),
                        "height": round(block['bounding_box']['height'])
                    }
                })
                block_id += 1
            
            page_to_blocks[page_num] = list(range(page_blocks_start, block_id))
        
        ocr_json_str = json.dumps(combined_blocks, indent=2)
        logger.info(f"📄 Combined OCR: {len(combined_blocks)} blocks from {len(all_ocr_data)} pages")
        
        prompt = self._get_multipage_prompt(ocr_json_str, len(combined_blocks), len(all_ocr_data))
        
        try:
            logger.info(f"🚀 Sending ALL pages ({len(all_ocr_data)} pages, {len(combined_blocks)} blocks) to LLM...")
            
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=ExtractedComponentData,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert aviation document analyzer processing MULTI-PAGE reports. "
                        "Each OCR block has a 'page' field indicating which page it's on. "
                        "For EVERY field you extract, you MUST: "
                        "1. Set the value in the appropriate field (TSN, CSN, SerialNumber, etc.) "
                        "2. Create a BoundingBox with 'page_number' set to the page where the field appears "
                        "3. Include the exact bounding box coordinates from that page's OCR data "
                        "Example: If Engine1 TSN '16300' appears on page 2 at coordinates (100, 200, 50, 20): "
                        "  Engine1.TSN = 16300 "
                        "  Engine1.TSN_bbox = BoundingBox(left=100, top=200, width=50, height=20, page_number=2) "
                        "Process ALL pages and ALL blocks. Extract ALL fields across ALL pages."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=16000
            )
            
            logger.info(f"✅ Extraction complete!")
            logger.info(f"\n{'='*60}")
            
            # Log extracted components
            component_names = ['Airframe', 'Engine1', 'Engine2', 'APU', 
                             'LandingGearLeft', 'LandingGearRight', 'LandingGearNose']
            
            for comp_name in component_names:
                comp_data = getattr(result, comp_name, None)
                if comp_data and comp_data != ComponentData():
                    # Check if component has any non-None values
                    has_data = any(
                        getattr(comp_data, field, None) is not None 
                        for field in ['SerialNumber', 'TSN', 'CSN', 'MonthlyUtil_Hrs', 
                                     'MonthlyUtil_Cyc', 'location', 'SerialNumber_Original']
                    )
                    
                    if has_data:
                        logger.info(f"🔧 {comp_name}:")
                        
                        # Log each field with its page number
                        for field_name in ['SerialNumber', 'SerialNumber_Original', 'TSN', 'CSN', 
                                         'MonthlyUtil_Hrs', 'MonthlyUtil_Cyc', 'location']:
                            value = getattr(comp_data, field_name, None)
                            bbox = getattr(comp_data, f"{field_name}_bbox", None)
                            
                            if value is not None:
                                page_num = bbox.page_number if bbox else "?"
                                logger.info(f"   {field_name}: {value} (Page {page_num})")
            
            logger.info(f"{'='*60}\n")
            
            # Convert to dict format for return
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"❌ Error in multi-page extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _get_multipage_prompt(self, ocr_json_str: str, total_blocks: int, total_pages: int) -> str:
        """Prompt for multi-page extraction using ExtractedComponentData structure"""
        
        return f"""Extract ALL component data from this MULTI-PAGE aviation report.

═══════════════════════════════════════════════════════════════════════════════
📊 MULTI-PAGE OCR DATA (ALL {total_pages} pages combined):
═══════════════════════════════════════════════════════════════════════════════

{ocr_json_str}

Total blocks: {total_blocks} from {total_pages} pages
Each block has: {{"id", "page", "text", "bbox"}}

═══════════════════════════════════════════════════════════════════════════════
🎯 CRITICAL: TRACK PAGE NUMBERS IN BOUNDING BOXES
═══════════════════════════════════════════════════════════════════════════════

For EVERY field extracted, you MUST:
1. Set the field value (e.g., TSN = 16300)
2. Create the corresponding _bbox field with:
   - left, top, width, height from the OCR block's bbox
   - **page_number** from the OCR block's "page" field

Example from OCR:
{{"id": 45, "page": 2, "text": "16300", "bbox": {{"left": 100, "top": 200, "width": 50, "height": 20}}}}

Extract as:
- Engine1.TSN = 16300
- Engine1.TSN_bbox = BoundingBox(left=100, top=200, width=50, height=20, page_number=2)

═══════════════════════════════════════════════════════════════════════════════
📋 EXTRACTION STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

**TERMINOLOGY VARIATIONS:**
- TSN = "Total Time Since New" = "Time Since New" = "TAH" = "Total Airframe Hours" = "Flight Hours"
- CSN = "Total Cycles Since New" = "Cycles Since New" = "TAC" = "Total Airframe Cycles" = "Flight Cycles"
- MonthlyUtil_Hrs = "HOURS FLOWN DURING MONTH" = "Delta Hrs" = "Period Hours" = "Period Airframe Hours"
- MonthlyUtil_Cyc = "CYCLES/LANDINGS DURING MONTH" = "Delta Cyc" = "Period Cycles" = "Period Airframe Cycles"

**1. Airframe** (Look for MSN, Aircraft Serial Number, Manufacturer Serial Number):
   Extract: SerialNumber, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc
   Each field with its _bbox (including page_number)

**2. Engine1** (Look for Position 1, 1000EM1, Engine 1, Left Engine):
   Extract: SerialNumber, SerialNumber_Original, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, location
   Each field with its _bbox (including page_number)

**3. Engine2** (Look for Position 2, 1000EM2, Engine 2, Right Engine):
   Extract: SerialNumber, SerialNumber_Original, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, location
   Each field with its _bbox (including page_number)

**4. APU** (Look for APU, Auxiliary Power Unit):
   Extract: SerialNumber, SerialNumber_Original, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, location
   Each field with its _bbox (including page_number)

**5. LandingGearLeft** (Look for Main Gear 1, Left Main Landing Gear):
   Extract: SerialNumber, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc
   Each field with its _bbox (including page_number)

**6. LandingGearRight** (Look for Main Gear 2, Right Main Landing Gear):
   Extract: SerialNumber, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc
   Each field with its _bbox (including page_number)

**7. LandingGearNose** (Look for Nose Gear, Nose Landing Gear):
   Extract: SerialNumber, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc
   Each field with its _bbox (including page_number)

═══════════════════════════════════════════════════════════════════════════════
✅ REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

1. ✓ Process ALL {total_blocks} blocks from ALL {total_pages} pages
2. ✓ For EVERY field, create both value AND _bbox fields
3. ✓ EVERY _bbox MUST include page_number from OCR block's "page" field
4. ✓ Extract ALL available fields for each component
5. ✓ If a component's data spans multiple pages, extract from ALL pages
6. ✓ Use None for fields that are not found

Return: ExtractedComponentData with all components populated"""