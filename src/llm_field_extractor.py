"""
Enhanced LLM-based field extraction using Azure OCR bounding boxes
OPTIMIZED VERSION: Single API call for discovery + extraction
Returns structured data using Pydantic models (ComponentData, ExtractedComponentData)
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
        Returns list of extraction results
        """
        logger.info("🔍 Starting unified extraction (single API call)...")
        
        text_blocks = ocr_results.get('text_blocks', [])
        simplified_blocks = []
        
        # Prepare simplified blocks with bounding box info
        for i, block in enumerate(text_blocks[:300]):
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
        
        # UNIFIED PROMPT: Discover + Extract in ONE call
        prompt = f"""Analyze this aviation document and extract ALL data in a SINGLE operation.

**OCR TEXT BLOCKS (with Azure OCR bounding boxes):**
{json.dumps(simplified_blocks[:150], indent=2)}

**STEP 1: IDENTIFY DOCUMENT TYPE**
Determine if this is:
- **component_data**: Contains TSN, CSN, MonthlyUtil for Airframe/Engines/APU/Landing Gear
- **standalone_assets**: Contains Month, MSN, ComponentSerialNumber, FlightRegistrationNumber
- **flight_info**: Contains Month, MSN, AirCraftType, RegistrationNumber

**STEP 2: FIND ALL IDENTIFIERS**
Identify ALL key identifiers (confidence > 0.8):
- aircraft_registration (e.g., VT-ABC, N12345, P-11217)
- engine_sn (Engine serial numbers)
- apu_sn (APU serial numbers)
- msn (Manufacturer Serial Number)
- component_sn (Any component serial number)

**STEP 3: EXTRACT DATA FOR EACH IDENTIFIER**
For EACH identifier found, extract the complete data based on document type:

**If component_data:**
Extract for each identifier in its column/section:
- Airframe: TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, SerialNumber, location (with block_id and bbox)
- Engine1: TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, SerialNumber, location, derate (with block_id and bbox)
- Engine2: TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, SerialNumber, location, derate (with block_id and bbox)
- APU: TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, SerialNumber, location (with block_id and bbox)
- LandingGearLeft: TSN, CSN, SerialNumber, location (with block_id and bbox)
- LandingGearRight: TSN, CSN, SerialNumber, location (with block_id and bbox)
- LandingGearNose: TSN, CSN, SerialNumber, location (with block_id and bbox)

**If standalone_assets:**
Extract: Month, MSN, ComponentSerialNumber, FlightRegistrationNumber (with block_id and bbox)

**If flight_info:**
Extract: Month, MSN, AirCraftType, RegistrationNumber (with block_id and bbox)

**CRITICAL:** For EVERY field, include the block_id from the OCR blocks where the value was found.

Return the complete structured extraction with ALL identifiers and their data in ONE response.
"""
        
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=CompleteDocumentExtraction,
                messages=[
                    {"role": "system", "content": (
        "You are an expert at analyzing Azure OCR JSON data. "
        "Each input contains text blocks with bounding boxes from aviation documents. "
        "Use this OCR data to extract all relevant fields and return them in a structured format."
    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=8192  # Increased for comprehensive extraction
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
        # Fallback to old behavior if needed
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
        # Fallback: extract all and filter
        results = self.extract_all_data(ocr_results)
        for r in results:
            if r['identifier'] == identifier:
                return r
        return {'found': False, 'message': 'Identifier not found'}