"""
Enhanced LLM-based field extraction using Azure OCR bounding boxes
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


class IdentifierLocation(BaseModel):
    """Location of an identifier in the document"""
    identifier: str = Field(description="The identifier text (e.g., serial number, registration)")
    identifier_type: str = Field(description="Type: aircraft_registration, engine_sn, apu_sn, msn, component_sn")
    block_id: int = Field(description="Block index in OCR results")
    confidence: float = Field(description="Confidence in identification (0-1)", ge=0, le=1)


class IdentifierDiscovery(BaseModel):
    """Result of identifier discovery"""
    found_identifiers: List[IdentifierLocation] = Field(
        description="All identifiers found in the document"
    )
    document_type: str = Field(
        description="Document type: component_data, standalone_assets, flight_info"
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
    
    def discover_identifiers(self, ocr_results: Dict[str, Any]) -> tuple[List[Dict[str, Any]], str]:
        """
        AUTO-DISCOVER all identifiers and determine document type
        Returns (identifiers, document_type)
        """
        logger.info("🔍 Auto-discovering identifiers and document type...")
        
        text_blocks = ocr_results.get('text_blocks', [])
        simplified_blocks = []
        
        for i, block in enumerate(text_blocks[:300]):
            simplified_blocks.append({
                "id": i,
                "text": block['text'],
                "x": round(block['bounding_box']['left']),
                "y": round(block['bounding_box']['top'])
            })
        
        prompt = f"""Analyze this aviation document and identify ALL key identifiers and document type.

**OCR TEXT BLOCKS (with Azure OCR bounding boxes):**
{json.dumps(simplified_blocks[:100], indent=2)}

**DOCUMENT TYPES:**
1. **component_data**: Contains TSN, CSN, MonthlyUtil for Airframe/Engines/APU/Landing Gear
2. **standalone_assets**: Contains Month, MSN, ComponentSerialNumber, FlightRegistrationNumber
3. **flight_info**: Contains Month, MSN, AirCraftType, RegistrationNumber

**IDENTIFIER TYPES:**
- aircraft_registration (e.g., VT-ABC, N12345, P-11217)
- engine_sn (Engine serial numbers)
- apu_sn (APU serial numbers)
- msn (Manufacturer Serial Number)
- component_sn (Any component serial number)

**TASK:**
1. Determine the document type based on the fields present
2. Find ALL identifiers in the document
3. For each identifier, provide the block_id where it appears

Return ALL identifiers with high confidence only (>0.8).
"""
        
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=IdentifierDiscovery,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing aviation maintenance documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2048
            )
            
            identifiers = []
            for id_loc in result.found_identifiers:
                identifiers.append({
                    'identifier': id_loc.identifier,
                    'type': id_loc.identifier_type,
                    'block_id': id_loc.block_id,
                    'confidence': id_loc.confidence
                })
            
            doc_type = result.document_type
            
            logger.info(f"✅ Document Type: {doc_type}")
            logger.info(f"✅ Found {len(identifiers)} identifiers")
            for idf in identifiers:
                logger.info(f"   - {idf['identifier']} ({idf['type']}, conf: {idf['confidence']:.2f})")
            
            return identifiers, doc_type
            
        except Exception as e:
            logger.error(f"❌ Error discovering identifiers: {str(e)}")
            return [], "unknown"
    
    def extract_all_data(self, ocr_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract structured data for ALL discovered identifiers using Pydantic models
        Returns list of extraction results
        """
        # Discover all identifiers and document type
        identifiers, doc_type = self.discover_identifiers(ocr_results)
        
        if not identifiers:
            logger.warning("⚠️ No identifiers found in document")
            return []
        
        # Extract data for each identifier
        all_results = []
        
        for idf in identifiers:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Extracting data for: {idf['identifier']} ({idf['type']})")
            logger.info(f"{'='*60}")
            
            try:
                result = self.extract_structured_data(
                    ocr_results, 
                    idf['identifier'], 
                    doc_type
                )
                
                if result['found']:
                    # Add identifier metadata
                    result['identifier_metadata'] = {
                        'type': idf['type'],
                        'confidence': idf['confidence'],
                        'block_id': idf['block_id']
                    }
                    result['document_type'] = doc_type
                    all_results.append(result)
                    logger.info(f"✅ Extracted {result.get('total_fields', 0)} fields")
                else:
                    logger.warning(f"⚠️ Could not extract data for {idf['identifier']}")
                    
            except Exception as e:
                logger.error(f"❌ Error extracting {idf['identifier']}: {str(e)}")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Total extractions: {len(all_results)}")
        logger.info(f"{'='*60}")
        
        return all_results
    
    def extract_structured_data(self, ocr_results: Dict[str, Any], 
                               identifier: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract data using appropriate Pydantic model based on document type
        Uses Azure OCR bounding boxes for precise field location
        """
        text_blocks = ocr_results.get('text_blocks', [])
        
        # Create simplified blocks with bounding box info
        simplified_blocks = []
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
        
        # Select appropriate extraction based on doc_type
        if doc_type == "component_data":
            return self._extract_component_data(simplified_blocks, identifier, text_blocks)
        elif doc_type == "standalone_assets":
            return self._extract_standalone_assets(simplified_blocks, identifier, text_blocks)
        elif doc_type == "flight_info":
            return self._extract_flight_info(simplified_blocks, identifier, text_blocks)
        else:
            # Fallback: try component data
            return self._extract_component_data(simplified_blocks, identifier, text_blocks)
    
    def _extract_component_data(self, simplified_blocks: List[Dict], 
                               identifier: str, text_blocks: List[Dict]) -> Dict[str, Any]:
        """Extract using ExtractedComponentData model"""
        
        prompt = f"""Extract component data for identifier "{identifier}" using the provided bounding boxes.

**OCR BLOCKS (from Azure OCR):**
{json.dumps(simplified_blocks[:150], indent=2)}

**TASK:**
Find "{identifier}" and extract ALL component data in the same column/section.

**COMPONENTS TO EXTRACT:**
- Airframe: TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, SerialNumber, location
- Engine1: TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, SerialNumber, location, derate
- Engine2: TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, SerialNumber, location, derate
- APU: TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, SerialNumber, location
- LandingGearLeft: TSN, CSN, SerialNumber, location
- LandingGearRight: TSN, CSN, SerialNumber, location
- LandingGearNose: TSN, CSN, SerialNumber, location

For each field, provide:
1. Value (extracted text)
2. block_id (the OCR block ID)

Return structured data with bounding boxes from Azure OCR.
"""
        
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=ExtractedComponentData,
                messages=[
                    {"role": "system", "content": "Extract component data using Azure OCR bounding boxes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=4096
            )
            
            # Convert Pydantic model to dict and enrich with bounding boxes
            extracted_dict = result.model_dump()
            enriched_data = self._enrich_with_bounding_boxes(extracted_dict, text_blocks)
            
            return {
                'found': True,
                'identifier': identifier,
                'identifier_type': 'component_data',
                'layout_type': 'columnar',
                'fields': enriched_data,
                'total_fields': self._count_fields(enriched_data),
                'document_type': 'component_data'
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting component data: {str(e)}")
            return {'found': False, 'message': str(e)}
    
    def _extract_standalone_assets(self, simplified_blocks: List[Dict],
                                   identifier: str, text_blocks: List[Dict]) -> Dict[str, Any]:
        """Extract using StandaloneAssetsData model"""
        
        prompt = f"""Extract standalone asset data for "{identifier}".

**OCR BLOCKS:**
{json.dumps(simplified_blocks[:100], indent=2)}

**FIELDS TO EXTRACT:**
- Month
- MSN
- ComponentSerialNumber
- FlightRegistrationNumber

Provide block_id for each field.
"""
        
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=StandaloneAssetsData,
                messages=[
                    {"role": "system", "content": "Extract standalone asset data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2048
            )
            
            extracted_dict = result.model_dump()
            enriched_data = self._enrich_with_bounding_boxes(extracted_dict, text_blocks)
            
            return {
                'found': True,
                'identifier': identifier,
                'identifier_type': 'standalone_assets',
                'fields': enriched_data,
                'total_fields': len(enriched_data),
                'document_type': 'standalone_assets'
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting standalone assets: {str(e)}")
            return {'found': False, 'message': str(e)}
    
    def _extract_flight_info(self, simplified_blocks: List[Dict],
                            identifier: str, text_blocks: List[Dict]) -> Dict[str, Any]:
        """Extract using FlightInfo model"""
        
        prompt = f"""Extract flight information for "{identifier}".

**OCR BLOCKS:**
{json.dumps(simplified_blocks[:100], indent=2)}

**FIELDS TO EXTRACT:**
- Month
- MSN
- AirCraftType
- RegistrationNumber

Provide block_id for each field.
"""
        
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=FlightInfo,
                messages=[
                    {"role": "system", "content": "Extract flight information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2048
            )
            
            extracted_dict = result.model_dump()
            enriched_data = self._enrich_with_bounding_boxes(extracted_dict, text_blocks)
            
            return {
                'found': True,
                'identifier': identifier,
                'identifier_type': 'flight_info',
                'fields': enriched_data,
                'total_fields': len(enriched_data),
                'document_type': 'flight_info'
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting flight info: {str(e)}")
            return {'found': False, 'message': str(e)}
    
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