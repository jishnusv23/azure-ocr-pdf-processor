"""
Enhanced LLM Field Extractor - COMPLETE MULTI-PAGE SUPPORT
Location: src/llm_field_extractor.py
"""
import json
import os
from typing import Dict, Any, List, Optional,Literal
import logging
from openai import OpenAI
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field

from src.model.extractor_model import (
    ComponentData, 
    BoundingBox, 
    ExtractedComponentData,
    StandaloneAssetsData,
    FlightInfo
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# MULTI-PAGE MODELS
# ============================================================================

class FieldWithPage(BaseModel):
    """Field value with its page number and bounding box"""
    field_name: Literal[
    "SerialNumber",
    "SerialNumber_Original", 
    "TSN",
    "CSN",
    "MonthlyUtil_Hrs",
    "MonthlyUtil_Cyc",
    "location"
] = Field(...)
    value: str = Field(description="Field value")
    page_number: int = Field(description="Page number where this field appears (1-indexed)")
    bounding_box: BoundingBox = Field(description="Bounding box coordinates")


class IdentifierWithPageData(BaseModel):
    """Identifier with ALL its fields organized by page"""
    identifier: str = Field(description="The identifier text")
    identifier_type: str = Field(description=("Component type: one of ['airframe', 'engine1', 'engine2', 'apu', ""'landing_gear_left', 'landing_gear_right', 'landing_gear_nose']"))
    confidence: float = Field(description="Confidence (0-1)", ge=0, le=1)
    fields: List[FieldWithPage] = Field(description="All fields with their page numbers and bounding boxes")


class MultiPageDocumentExtraction(BaseModel):
    """Complete extraction with page-aware field tracking"""
    document_type: str = Field(description="Document type: component_data, standalone_assets, flight_info")
    total_pages: int = Field(description="Total pages processed")
    identifiers: List[IdentifierWithPageData] = Field(description="All identifiers with page-aware fields")


# ============================================================================
# LEGACY MODELS (for backward compatibility)
# ============================================================================

class IdentifierWithData(BaseModel):
    """Complete identifier with its extracted data"""
    identifier: str = Field(description="The identifier text")
    identifier_type: str = Field(description="Type: aircraft_registration, engine_sn, apu_sn, msn, component_sn")
    confidence: float = Field(description="Confidence (0-1)", ge=0, le=1)
    component_data: Optional[ExtractedComponentData] = None
    standalone_assets: Optional[StandaloneAssetsData] = None
    flight_info: Optional[FlightInfo] = None


class CompleteDocumentExtraction(BaseModel):
    """Single unified response with all identifiers and their data"""
    document_type: str = Field(
        description="Document type: component_data, standalone_assets, flight_info"
    )
    identifiers_with_data: List[IdentifierWithData] = Field(
        description="All identifiers found with their complete extracted data"
    )


# ============================================================================
# LLM FIELD EXTRACTOR
# ============================================================================

class LLMFieldExtractor:
    """LLM Field Extractor - Multi-Page Support for ALL aviation PDF formats"""
    
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
                                   pdf_filename: str) -> List[Dict[str, Any]]:
        """
        MULTI-PAGE EXTRACTION: Process ALL pages together
        Returns identifiers with fields organized by page number
        
        Args:
            all_ocr_data: List of OCR data dicts, one per page
            pdf_filename: Name of PDF file
            
        Returns:
            List of dicts with structure:
            {
                'identifier': 'ENGINE_SN',
                'identifier_type': 'engine_sn',
                'fields_by_page': {
                    1: [{'field': 'SerialNumber', 'value': '862909', 'bounding_box': {...}}],
                    2: [{'field': 'TSN', 'value': '16300', 'bounding_box': {...}}]
                }
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
                response_model=MultiPageDocumentExtraction,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert aviation document analyzer processing MULTI-PAGE reports. "
                        "Each OCR block has a 'page' field indicating which page it's on. "
                        "For EVERY field you extract, you MUST: "
                        "1. Include the page number where the field appears "
                        "2. Include the exact bounding box from that page's OCR data "
                        "3. Track which page each field is on (page numbers are in the OCR data) "
                        "Example: If Engine TSN appears on page 2, record: page_number=2, bbox from page 2 "
                        "Process ALL pages and ALL blocks. Extract ALL fields for each identifier across ALL pages."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=16000
            )
            
            logger.info(f"✅ Document Type: {result.document_type}")
            logger.info(f"✅ Found {len(result.identifiers)} identifiers across {result.total_pages} pages")
            
            # Organize results by identifier with fields grouped by page
            final_results = []
            
            for id_data in result.identifiers:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔍 Identifier: {id_data.identifier} ({id_data.identifier_type})")
                
                # Group fields by page
                fields_by_page = {}
                for field in id_data.fields:
                    page_num = field.page_number
                    if page_num not in fields_by_page:
                        fields_by_page[page_num] = []
                    
                    fields_by_page[page_num].append({
                        'field': field.field_name,
                        'value': field.value,
                        'bounding_box': field.bounding_box.model_dump()
                    })
                
                logger.info(f"   📄 Fields found on pages: {sorted(fields_by_page.keys())}")
                for page_num, fields in sorted(fields_by_page.items()):
                    logger.info(f"      Page {page_num}: {len(fields)} fields")
                
                final_results.append({
                    'identifier': id_data.identifier,
                    'identifier_type': id_data.identifier_type,
                    'confidence': id_data.confidence,
                    'fields_by_page': fields_by_page,
                    'total_fields': len(id_data.fields)
                })
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🎉 Multi-page extraction complete")
            logger.info(f"✅ Identifiers: {len(final_results)}")
            logger.info(f"✅ Pages processed: {len(all_ocr_data)}")
            logger.info(f"{'='*60}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error in multi-page extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _get_multipage_prompt(self, ocr_json_str: str, total_blocks: int, total_pages: int) -> str:
        """Prompt for multi-page extraction"""
        
        return f"""Extract ALL identifiers and their complete data from this MULTI-PAGE aviation report.

═══════════════════════════════════════════════════════════════════════════════
📊 MULTI-PAGE OCR DATA (ALL {total_pages} pages combined):
═══════════════════════════════════════════════════════════════════════════════

{ocr_json_str}

Total blocks: {total_blocks} from {total_pages} pages
Each block has: {{"id", "page", "text", "bbox"}}

═══════════════════════════════════════════════════════════════════════════════
🎯 CRITICAL: TRACK PAGE NUMBERS
═══════════════════════════════════════════════════════════════════════════════

For EVERY field extracted, you MUST record:
1. **field_name**: The field name (e.g., "TSN", "CSN", "SerialNumber")
2. **value**: The field value
3. **page_number**: Which page this field appears on (from the "page" key in OCR block)
4. **bounding_box**: The exact bbox from that page's OCR block

Example:
- OCR block: {{"id": 45, "page": 2, "text": "16300", "bbox": {{...}}}}
- Extract as: field_name="TSN", value="16300", page_number=2, bounding_box={{...}}

═══════════════════════════════════════════════════════════════════════════════
📋 EXTRACTION TASK
═══════════════════════════════════════════════════════════════════════════════

**STEP 1: Find ALL identifiers** (scan ALL {total_blocks} blocks across ALL pages)
- msn: "MSN", "Serialnumber" (e.g., 9999, 02607, 3184)
- aircraft_registration: "REGISTRATION", "Current Registration" (e.g., A-7575, AKNT, G-EZBZ)
- engine_sn: "S/N of Engine", "ESN", "Serialnumber" in engine section (e.g., 862909, 779682, 643464)
- apu_sn: APU serial, "Serialnumber" in APU section (e.g., P-11217, P-3775, P-2882)
- component_sn: Landing gear (e.g., MDG1233, B3219, MDL-2946)

**STEP 2: Extract ALL fields for each identifier across ALL pages**

**TERMINOLOGY VARIATIONS:**
- TSN = "Total Time Since New" = "Time Since New" = "TAH" = "Total Airframe Hours" = "Flight Hours"
- CSN = "Total Cycles Since New" = "Cycles Since New" = "TAC" = "Total Airframe Cycles" = "Flight Cycles"
- MonthlyUtil_Hrs = "HOURS FLOWN DURING MONTH" = "Delta Hrs" = "Period Hours" = "Period Airframe Hours"
- MonthlyUtil_Cyc = "CYCLES/LANDINGS DURING MONTH" = "Delta Cyc" = "Period Cycles" = "Period Airframe Cycles"

**For AIRFRAME (MSN identifier):**
Extract: SerialNumber, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc
(Track which page each field appears on!)

**For ENGINES (Position 1, Position 2, 1000EM1, 1000EM2):**
Extract: SerialNumber, SerialNumber_Original, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, location
(Engine data may span multiple pages - track each field's page!)

**For APU:**
Extract: SerialNumber, SerialNumber_Original, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, location
(APU data may span multiple pages - track each field's page!)

**For LANDING GEAR (Left, Right, Nose, Main Gear 1, Main Gear 2):**
Extract: SerialNumber, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc
(Track which page each field appears on!)

═══════════════════════════════════════════════════════════════════════════════
✅ REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

1. ✓ Process ALL {total_blocks} blocks from ALL {total_pages} pages
2. ✓ For EVERY field, include: field_name, value, page_number, bounding_box
3. ✓ Page numbers come from the "page" key in OCR blocks
4. ✓ Extract ALL fields (not just SerialNumber) for each identifier
5. ✓ If an identifier's data spans multiple pages, extract from ALL pages

Return: MultiPageDocumentExtraction with page-aware field tracking"""
    
    # ============================================================================
    # LEGACY METHOD (for backward compatibility)
    # ============================================================================
    
    def extract_all_data(self, ocr_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        LEGACY: Single-page extraction (kept for backward compatibility)
        For multi-page PDFs, use extract_all_data_multipage() instead
        """
        logger.warning("⚠️ Using legacy single-page extraction. Use extract_all_data_multipage() for better multi-page support.")
        
        text_blocks = ocr_results.get('text_blocks', [])
        logger.info(f"📊 Total OCR blocks: {len(text_blocks)}")
        
        # Prepare simplified OCR blocks
        simplified_blocks = []
        for i, block in enumerate(text_blocks):
            simplified_blocks.append({
                "id": i,
                "text": block['text'],
                "bbox": {
                    "left": round(block['bounding_box']['left']),
                    "top": round(block['bounding_box']['top']),
                    "width": round(block['bounding_box']['width']),
                    "height": round(block['bounding_box']['height'])
                }
            })
        
        ocr_json_str = json.dumps(simplified_blocks, indent=2)
        prompt = self._get_universal_prompt(ocr_json_str, len(text_blocks))
        
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=CompleteDocumentExtraction,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert aviation document analyzer with deep knowledge of ALL aircraft utilization report formats. "
                        "You process ALL OCR blocks without skipping. "
                        "For EVERY field extracted, you include the exact bounding box from the OCR data. "
                        "When you find an identifier (MSN, Engine S/N, etc.), you extract ALL associated fields, not just the serial number."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=16000
            )
            
            # Convert to output format
            all_results = []
            
            for id_data in result.identifiers_with_data:
                extracted_dict = None
                if id_data.component_data:
                    extracted_dict = id_data.component_data.model_dump()
                elif id_data.standalone_assets:
                    extracted_dict = id_data.standalone_assets.model_dump()
                elif id_data.flight_info:
                    extracted_dict = id_data.flight_info.model_dump()
                
                if extracted_dict:
                    enriched_data = self._enrich_with_bounding_boxes(extracted_dict, text_blocks)
                    
                    result_entry = {
                        'found': True,
                        'identifier': id_data.identifier,
                        'identifier_type': id_data.identifier_type,
                        'identifier_metadata': {
                            'type': id_data.identifier_type,
                            'confidence': id_data.confidence
                        },
                        'document_type': result.document_type,
                        'fields': enriched_data,
                        'total_fields': self._count_fields(enriched_data)
                    }
                    all_results.append(result_entry)
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Error in extraction: {str(e)}")
            return []
    
    def _get_universal_prompt(self, ocr_json_str: str, total_blocks: int) -> str:
        """Universal prompt for single-page extraction"""
        
        return f"""Analyze this aviation utilization report and extract ALL data.

{ocr_json_str}

Total blocks: {total_blocks}

Find ALL identifiers (msn, aircraft_registration, engine_sn, apu_sn, component_sn) and extract their COMPLETE data.
For each identifier, extract ALL fields: SerialNumber, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, location.
Include bounding boxes for ALL fields from the OCR data.

Terminology: TSN = Total Time Since New = TAH, CSN = Total Cycles Since New = TAC"""
    
    def _enrich_with_bounding_boxes(self, extracted_dict: Dict[str, Any], 
                                    text_blocks: List[Dict]) -> Dict[str, Any]:
        """Add Azure OCR bounding boxes to extracted data"""
        enriched = {}
        
        for key, value in extracted_dict.items():
            if value is None:
                continue
            
            if isinstance(value, dict):
                component_enriched = {}
                for field_name, field_value in value.items():
                    if field_value is not None and not field_name.endswith('_bbox'):
                        bbox_key = f"{field_name}_bbox"
                        if bbox_key in value and value[bbox_key]:
                            bbox_data = value[bbox_key]
                            if hasattr(bbox_data, 'dict'):
                                bbox_dict = bbox_data.dict()
                            elif hasattr(bbox_data, 'model_dump'):
                                bbox_dict = bbox_data.model_dump()
                            else:
                                bbox_dict = bbox_data
                            
                            component_enriched[field_name] = {
                                'value': field_value,
                                'bounding_box': bbox_dict
                            }
                        else:
                            component_enriched[field_name] = {'value': field_value}
                
                if component_enriched:
                    enriched[key] = component_enriched
            else:
                enriched[key] = {'value': value}
        
        return enriched
    
    def _count_fields(self, data: Dict[str, Any]) -> int:
        """Count total extracted fields"""
        count = 0
        for value in data.values():
            if isinstance(value, dict):
                count += len([v for v in value.values() if v is not None])
            elif value is not None:
                count += 1
        return count