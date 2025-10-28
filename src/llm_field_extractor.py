"""
Enhanced LLM-based field extraction using Azure OCR bounding boxes
FIXED: MSN (9999) now extracts COMPLETE Airframe data (TSN, CSN, MonthlyUtil, etc.)
"""
import json
import os
from typing import Dict, Any, List, Optional
import logging
from openai import OpenAI
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field

# Import your extraction models
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


class IdentifierWithData(BaseModel):
    """Complete identifier with its extracted data"""
    identifier: str = Field(description="The identifier text")
    identifier_type: str = Field(description="Type: aircraft_registration, engine_sn, apu_sn, msn, component_sn")
    confidence: float = Field(description="Confidence (0-1)", ge=0, le=1)
    # Data fields based on document type
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


class LLMFieldExtractor:
    """LLM Field Extractor using Azure OCR bounding boxes and Pydantic models"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        
        if not self.api_key:
            raise ValueError("❌ OPENROUTER_API_KEY not set in .env file")
        
        base_client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.client = instructor.from_openai(base_client)
        logger.info(f"✅ LLM Field Extractor initialized (Model: {self.model})")
    
    def extract_all_data(self, ocr_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Single API call to discover identifiers AND extract all data
        FIXED: MSN now extracts COMPLETE Airframe data (TSN, CSN, MonthlyUtil, etc.)
        """
        logger.info("🔍 Starting unified extraction (single API call)...")
        
        # Prepare ALL text blocks (no truncation)
        text_blocks = ocr_results.get('text_blocks', [])
        logger.info(f"📊 Total OCR blocks: {len(text_blocks)} (processing ALL blocks)")
        
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
        
        # Create prompt with COMPLETE OCR data
        ocr_json_str = json.dumps(simplified_blocks, indent=2)
        logger.info(f"📄 OCR JSON size: {len(ocr_json_str)} characters")
        
        prompt = f"""Analyze this aviation utilization report and extract ALL data in a SINGLE operation.

**COMPLETE OCR TEXT BLOCKS (with Azure OCR bounding boxes):**
{ocr_json_str}

**STEP 1: IDENTIFY DOCUMENT TYPE**
Determine if this is:
- **component_data**: Contains component utilization data (TSN, CSN, hours, cycles for aircraft components)
- **standalone_assets**: Contains standalone component data with Month, MSN, ComponentSerialNumber
- **flight_info**: Contains flight/aircraft information with Month, MSN, AirCraftType, RegistrationNumber

**STEP 2: FIND ALL IDENTIFIERS**
Scan through ALL {len(text_blocks)} OCR blocks to identify ALL key identifiers (confidence > 0.8):
- **aircraft_registration**: Aircraft registration numbers (e.g., VT-ABC, N12345, A-7575, AKNT, AKNU, AKNV, OO-SSJ)
- **engine_sn**: Engine serial numbers (e.g., 862909, 779682, 577611)
- **apu_sn**: APU serial numbers (e.g., P-11217, P-3775, P-3067, P-4503, P-4431)
- **msn**: Manufacturer Serial Number (e.g., 9999, 02607, 02628, 02632, 1759)
- **component_sn**: Component serial numbers (e.g., MDG1233, B3219, M-DG-1967-010)
- **esn**: Engine Serial Number from standalone reports

**STEP 3: EXTRACT DATA FOR EACH IDENTIFIER**
For EACH identifier found, extract ALL available data based on document type and format:

**CRITICAL FOR MSN IDENTIFIERS:**
When you find an MSN identifier (e.g., "9999"), you MUST extract the COMPLETE Airframe component data:
- Airframe.SerialNumber: The MSN value itself (with bounding box)
- Airframe.TSN: "AIRCRAFT TOTAL TIME SINCE NEW" or "TAH" (with bounding box)
- Airframe.CSN: "TOTAL CYCLES SINCE NEW" or "TAC" (with bounding box)
- Airframe.MonthlyUtil_Hrs: "HOURS FLOWN DURING MONTH" (with bounding box)
- Airframe.MonthlyUtil_Cyc: "CYCLES/LANDINGS DURING MONTH" (with bounding box)

DO NOT extract ONLY Airframe.SerialNumber for MSN - extract ALL Airframe fields!

**If component_data (handles ALL util report formats):**

For Airframe (EXTRACT ALL FIELDS, not just SerialNumber):
- **SerialNumber**: Extract the **MSN (Manufacturer Serial Number)** value
  - Look for "MSN:", "Manufacturer Serial No.", "M.S.N.", or similar labels
  - Find the OCR block containing the MSN VALUE (e.g., "9999")
  - Extract the bounding box from that OCR block (use the block's id and bbox)
- **TSN**: "AIRCRAFT TOTAL TIME SINCE NEW" or "TAH" (with bounding box)
- **CSN**: "TOTAL CYCLES SINCE NEW" or "TAC" (with bounding box)
- **MonthlyUtil_Hrs**: "HOURS FLOWN DURING MONTH" (with bounding box)
- **MonthlyUtil_Cyc**: "CYCLES/LANDINGS DURING MONTH" (with bounding box)

For each Engine Position (1, 2, etc.) - EXTRACT ALL FIELDS:
- **SerialNumber**: Value from "S/N of Engine Installed" field (PRIMARY serial number)
- **SerialNumber_Original**: Value from "S/N of Original Engine" or "S/N of Original Engine's" field
- **TSN**: "Total Time Since New of Original Engine" or "Total Time Since New"
- **CSN**: "Total Cycles Since New of Original Engine" or "Total Cycles Since New"
- **MonthlyUtil_Hrs**: "Hours flown during Month of Original Engine" or "Hours flown during Month"
- **MonthlyUtil_Cyc**: "Cycles During Month of Original Engine" or "Cycles During Month"
- **location**: "Present Location of Original Engine" or "Present Location"

For APU - EXTRACT ALL FIELDS:
- **SerialNumber**: From "S/N of Engine Installed" or APU serial number field
- **SerialNumber_Original**: From "S/N of Original Engine" or "S/N of Original Engine's"
- **TSN**, **CSN**, **MonthlyUtil_Hrs**, **MonthlyUtil_Cyc**, **location**

For Landing Gear (Left/Main 1, Right/Main 2, Nose) - EXTRACT ALL FIELDS:
- **SerialNumber**: From "S/N of Landing Gear Installed"
- **TSN**: "Total Time Since New"
- **CSN**: "Total Cycles Since New"
- **MonthlyUtil_Hrs**: "Total Hours Flown During Month"
- **MonthlyUtil_Cyc**: "Total Cycles Made During Month"

**CRITICAL BOUNDING BOX EXTRACTION:**
1. For EVERY field value extracted, find the OCR block containing that value
2. Use the block's "id" and "bbox" from the OCR JSON
3. Include the bounding box in the extraction result
4. Example: If Airframe.TSN = "56748.23", find the block where text="56748.23" and use its bbox
5. DO NOT extract fields without bounding boxes unless the value is computed or unavailable in OCR

**If standalone_assets:**
Extract: Month, MSN, ComponentSerialNumber, FlightRegistrationNumber, AircraftType
(with bounding boxes from OCR blocks)

**If flight_info:**
Extract: Month, MSN, AirCraftType, RegistrationNumber
(with bounding boxes from OCR blocks)

**EXTRACTION RULES:**
1. Process ALL {len(text_blocks)} text blocks
2. For EVERY field extracted, include the block_id and bounding box from OCR
3. For identifiers like MSN, engine_sn, apu_sn, component_sn: Extract the COMPLETE component data (all TSN, CSN, MonthlyUtil fields)
4. DO NOT create partial extractions with only SerialNumber - extract ALL available fields for that component
5. If you cannot find a bounding box for a field, check if it's in the OCR blocks

Return the complete structured extraction with ALL identifiers and their COMPLETE component data in ONE response.
"""
        
        try:
            logger.info(f"🚀 Sending complete OCR ({len(text_blocks)} blocks) to LLM...")
            
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=CompleteDocumentExtraction,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert at analyzing complete Azure OCR JSON data. "
                        "Process ALL text blocks provided - do not skip any blocks. "
                        "For EVERY field extracted, you MUST include the bounding box from the OCR blocks. "
                        "Find the exact OCR block containing each field value and extract its bbox coordinates. "
                        "When extracting component data (Airframe, Engine, APU, Landing Gear), extract ALL fields "
                        "(SerialNumber, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc) not just the serial number. "
                        "This is critical for PDF highlighting to work correctly."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=16000
            )
            
            doc_type = result.document_type
            logger.info(f"✅ Document Type: {doc_type}")
            logger.info(f"✅ Found {len(result.identifiers_with_data)} identifiers with data")
            
            # Convert to output format
            all_results = []
            
            for id_data in result.identifiers_with_data:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔍 Identifier: {id_data.identifier} ({id_data.identifier_type})")
                logger.info(f"   Confidence: {id_data.confidence:.2f}")
                
                # Determine which data was extracted
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
                        'document_type': doc_type,
                        'fields': enriched_data,
                        'total_fields': self._count_fields(enriched_data)
                    }
                    
                    all_results.append(result_entry)
                    logger.info(f"✅ Extracted {result_entry['total_fields']} fields")
                else:
                    logger.warning(f"⚠️ No data extracted for {id_data.identifier}")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🎉 Total extractions: {len(all_results)}")
            logger.info(f"✅ Processed ALL {len(text_blocks)} OCR blocks")
            logger.info(f"✅ API Calls Made: 1 (instead of {len(all_results) + 1})")
            logger.info(f"{'='*60}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Error in unified extraction: {str(e)}")
            return []
    
    def _enrich_with_bounding_boxes(self, extracted_dict: Dict[str, Any], 
                                    text_blocks: List[Dict]) -> Dict[str, Any]:
        """Add Azure OCR bounding boxes to extracted data"""
        enriched = {}
        
        for key, value in extracted_dict.items():
            if value is None:
                continue
            
            # Handle nested ComponentData objects
            if isinstance(value, dict):
                component_enriched = {}
                for field_name, field_value in value.items():
                    if field_value is not None and not field_name.endswith('_bbox'):
                        # Check if there's a corresponding bbox field
                        bbox_key = f"{field_name}_bbox"
                        if bbox_key in value and value[bbox_key]:
                            bbox_data = value[bbox_key]
                            # Convert BoundingBox model to dict if needed
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

    def discover_identifiers(self, ocr_results: Dict[str, Any]) -> tuple[List[Dict[str, Any]], str]:
        """
        DEPRECATED: Use extract_all_data() instead for single API call
        This method is kept for backwards compatibility
        """
        logger.warning("⚠️ discover_identifiers() is deprecated. Use extract_all_data() for optimized single API call.")
        results = self.extract_all_data(ocr_results)
        identifiers = [
            {
                'identifier': r['identifier'],
                'type': r['identifier_type'],
                'confidence': r['identifier_metadata']['confidence']
            }
            for r in results
        ]
        doc_type = results[0]['document_type'] if results else "unknown"
        return identifiers, doc_type
    
    def extract_structured_data(self, ocr_results: Dict[str, Any], 
                               identifier: str, doc_type: str) -> Dict[str, Any]:
        """
        DEPRECATED: Use extract_all_data() instead for single API call
        This method is kept for backwards compatibility
        """
        logger.warning("⚠️ extract_structured_data() is deprecated. Use extract_all_data() for optimized single API call.")
        results = self.extract_all_data(ocr_results)
        for r in results:
            if r['identifier'] == identifier:
                return r
        return {'found': False, 'message': 'Identifier not found'}