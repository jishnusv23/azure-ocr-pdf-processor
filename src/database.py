"""
Database operations for storing OCR results using Pydantic models
FIXED: Properly extracts plain values from {'value': ..., 'bounding_box': ...} structure
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import json

from config.config import DATABASE_URL


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class Document(Base):
    """Documents (PDF files)"""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    total_pages = Column(Integer)
    processed_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    ocr_results = relationship("OCRResult", back_populates="document", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('filename', 'file_path', name='uq_document_filename_path'),
    )


class OCRResult(Base):
    """OCR Results (per page)"""
    __tablename__ = 'ocr_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    page_number = Column(Integer, nullable=False)
    total_blocks = Column(Integer)
    image_width = Column(Float)
    image_height = Column(Float)
    ocr_json = Column(Text, nullable=False)
    processed_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="ocr_results")
    identifiers = relationship("Identifier", back_populates="ocr_result", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('document_id', 'page_number', name='uq_ocr_document_page'),
    )


class Identifier(Base):
    """Identifiers (discovered serial numbers, registrations, etc.)"""
    __tablename__ = 'identifiers'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ocr_result_id = Column(Integer, ForeignKey('ocr_results.id'), nullable=False)
    identifier = Column(String, nullable=False)
    identifier_type = Column(String, nullable=False)
    confidence = Column(Float)
    block_id = Column(Integer)
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    ocr_result = relationship("OCRResult", back_populates="identifiers")
    extracted_data = relationship("ExtractedData", back_populates="identifier", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('ocr_result_id', 'identifier', name='uq_identifier_ocr_identifier'),
    )


class ExtractedData(Base):
    """Extracted Data (LLM extraction results)"""
    __tablename__ = 'extracted_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    identifier_id = Column(Integer, ForeignKey('identifiers.id'), nullable=False)
    document_type = Column(String)
    layout_type = Column(String)
    extraction_json = Column(Text, nullable=False)
    total_fields = Column(Integer)
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    identifier = relationship("Identifier", back_populates="extracted_data")
    components = relationship("ComponentTable", back_populates="extracted_data", cascade="all, delete-orphan")
    standalone_assets = relationship("StandaloneAssetTable", back_populates="extracted_data", cascade="all, delete-orphan")
    flight_info = relationship("FlightInfoTable", back_populates="extracted_data", uselist=False, cascade="all, delete-orphan")


class ComponentTable(Base):
    __tablename__ = 'components'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    extracted_data_id = Column(Integer, ForeignKey('extracted_data.id'), nullable=False)
    component_type = Column(String, nullable=False)
    
    
    
    tsn = Column(Float)
    tsn_bbox_json = Column(JSONB)
    
    csn = Column(Integer)
    csn_bbox_json = Column(JSONB)
    
    monthly_util_hrs = Column(Float)
    monthly_util_hrs_bbox_json = Column(JSONB)
    
    monthly_util_cyc = Column(Integer)
    monthly_util_cyc_bbox_json = Column(JSONB)
    
    serial_number = Column(String)
    serial_number_bbox_json = Column(JSONB)

    serial_number_original = Column(String)
    serial_number_original_bbox_json = Column(JSONB)
    
    location = Column(String)
    location_bbox_json = Column(JSONB)
    
    derate = Column(Float)
    extraction_confidence = Column(Float)
    created_date = Column(DateTime, default=datetime.utcnow)
    
    extracted_data = relationship("ExtractedData", back_populates="components")


class StandaloneAssetTable(Base):
    """Standalone Assets Data"""
    __tablename__ = 'standalone_assets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    extracted_data_id = Column(Integer, ForeignKey('extracted_data.id'), nullable=False)
    
    month = Column(String)
    msn = Column(String)
    flight_registration_number = Column(String)
    component_serial_number = Column(String, nullable=False)
    
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    extracted_data = relationship("ExtractedData", back_populates="standalone_assets")


class FlightInfoTable(Base):
    """Flight Information Data"""
    __tablename__ = 'flight_information'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    extracted_data_id = Column(Integer, ForeignKey('extracted_data.id'), nullable=False)
    
    month = Column(String)
    msn = Column(String)
    aircraft_type = Column(String)
    registration_number = Column(String)
    
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    extracted_data = relationship("ExtractedData", back_populates="flight_info")


class DatabaseManager:
    """Manages database operations for OCR and extraction results using SQLAlchemy and Pydantic"""
    
    def __init__(self, database_url: str = DATABASE_URL):
        """Initialize database connection"""
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.initialize_database()
    
    def initialize_database(self):
        """Create tables if they don't exist"""
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"✅ Database initialized: {self.engine.url}")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_document(self, filename: str, file_path: str, total_pages: int) -> int:
        """Save document metadata"""
        with self.get_session() as session:
            try:
                existing_doc = session.query(Document).filter(
                    Document.filename == filename,
                    Document.file_path == file_path
                ).first()
                
                if existing_doc:
                    return existing_doc.id
                
                document = Document(
                    filename=filename,
                    file_path=file_path,
                    total_pages=total_pages
                )
                session.add(document)
                session.commit()
                session.refresh(document)
                return document.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving document: {e}")
                raise
    
    def save_ocr_result(self, document_id: int, page_number: int, 
                       ocr_results: Dict[str, Any]) -> int:
        """Save OCR results for a page"""
        with self.get_session() as session:
            try:
                existing_ocr = session.query(OCRResult).filter(
                    OCRResult.document_id == document_id,
                    OCRResult.page_number == page_number
                ).first()
                
                img_dims = ocr_results.get('image_dimensions', {})
                
                if existing_ocr:
                    existing_ocr.total_blocks = len(ocr_results.get('text_blocks', []))
                    existing_ocr.image_width = img_dims.get('width')
                    existing_ocr.image_height = img_dims.get('height')
                    existing_ocr.ocr_json = json.dumps(ocr_results)
                    session.commit()
                    return existing_ocr.id
                else:
                    ocr_result = OCRResult(
                        document_id=document_id,
                        page_number=page_number,
                        total_blocks=len(ocr_results.get('text_blocks', [])),
                        image_width=img_dims.get('width'),
                        image_height=img_dims.get('height'),
                        ocr_json=json.dumps(ocr_results)
                    )
                    session.add(ocr_result)
                    session.commit()
                    session.refresh(ocr_result)
                    return ocr_result.id
                    
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving OCR result: {e}")
                raise
    
    def save_identifier(self, ocr_result_id: int, identifier: str, 
                       identifier_type: str, confidence: float, block_id: int) -> int:
        """Save discovered identifier"""
        with self.get_session() as session:
            try:
                existing_identifier = session.query(Identifier).filter(
                    Identifier.ocr_result_id == ocr_result_id,
                    Identifier.identifier == identifier
                ).first()
                
                if existing_identifier:
                    return existing_identifier.id
                
                identifier_obj = Identifier(
                    ocr_result_id=ocr_result_id,
                    identifier=identifier,
                    identifier_type=identifier_type,
                    confidence=confidence,
                    block_id=block_id
                )
                session.add(identifier_obj)
                session.commit()
                session.refresh(identifier_obj)
                return identifier_obj.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving identifier: {e}")
                raise
    
    def save_extraction_result(self, identifier_id: int, 
                               extraction_result: Dict[str, Any]) -> int:
        """Save LLM extraction results with Pydantic model support"""
        with self.get_session() as session:
            try:
                # Save main extraction data
                extracted_data = ExtractedData(
                    identifier_id=identifier_id,
                    document_type=extraction_result.get('document_type'),
                    layout_type=extraction_result.get('layout_type'),
                    extraction_json=json.dumps(extraction_result),
                    total_fields=extraction_result.get('total_fields', 0)
                )
                session.add(extracted_data)
                session.commit()
                session.refresh(extracted_data)
                
                # Parse extraction result based on document type
                doc_type = extraction_result.get('document_type', '').lower()
                
                if 'component' in doc_type:
                    # Parse as ExtractedComponentData
                    self._save_component_data(session, extracted_data.id, extraction_result)
                
                elif 'standalone' in doc_type:
                    # Parse as StandaloneAssetsData
                    self._save_standalone_assets(session, extracted_data.id, extraction_result)
                
                elif 'flight' in doc_type:
                    # Parse as FlightInfo
                    self._save_flight_info(session, extracted_data.id, extraction_result)
                
                session.commit()
                return extracted_data.id
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving extraction result: {e}")
                raise
    
    def _extract_value_from_field(self, field_data: Any) -> Any:
        """
        Extract plain value from field data structure
        Handles both {'value': X, 'bounding_box': Y} and plain values
        """
        if field_data is None:
            return None
        
        # If it's a dict with 'value' key, extract the value
        if isinstance(field_data, dict) and 'value' in field_data:
            return field_data['value']
        
        # Otherwise return as-is
        return field_data
    
    def _extract_bbox_from_field(self, field_data: Any) -> Optional[Dict]:
        """
        Extract bounding box from field data structure
        Returns None if no bounding box found
        """
        if field_data is None:
            return None
        
        # If it's a dict with 'bounding_box' key, extract it
        if isinstance(field_data, dict) and 'bounding_box' in field_data:
            return field_data['bounding_box']
        
        return None
    
    def _save_component_data(self, session, extracted_data_id: int, extraction_result: Dict[str, Any]):
        """Save component data from ExtractedComponentData model"""
        try:
            fields = extraction_result.get('fields', {})
            
            # Component types from ExtractedComponentData
            component_types = ['Airframe', 'Engine1', 'Engine2', 'APU', 
                             'LandingGearLeft', 'LandingGearRight', 'LandingGearNose']
            
            for comp_type in component_types:
                comp_data = fields.get(comp_type)
                if comp_data and isinstance(comp_data, dict):
                    # Extract values and bboxes separately
                    tsn_data = comp_data.get('TSN')
                    csn_data = comp_data.get('CSN')
                    monthly_hrs_data = comp_data.get('MonthlyUtil_Hrs')
                    monthly_cyc_data = comp_data.get('MonthlyUtil_Cyc')
                    serial_data = comp_data.get('SerialNumber')
                    serial_original_data = comp_data.get('SerialNumber_Original')
                    location_data = comp_data.get('location')
                    derate_data = comp_data.get('derate')
                    
                    component = ComponentTable(
                        extracted_data_id=extracted_data_id,
                        component_type=comp_type,
                        # Extract plain values
                        tsn=self._to_float(self._extract_value_from_field(tsn_data)),
                        csn=self._to_int(self._extract_value_from_field(csn_data)),
                        monthly_util_hrs=self._to_float(self._extract_value_from_field(monthly_hrs_data)),
                        monthly_util_cyc=self._to_int(self._extract_value_from_field(monthly_cyc_data)),
                        serial_number=self._to_str(self._extract_value_from_field(serial_data)),
                        serial_number_original=self._to_str(self._extract_value_from_field(serial_original_data)),
                        location=self._to_str(self._extract_value_from_field(location_data)),
                        derate=self._to_float(self._extract_value_from_field(derate_data)),
                        # Extract bounding boxes as JSON
                        tsn_bbox_json=self._extract_bbox_from_field(tsn_data),
                        csn_bbox_json=self._extract_bbox_from_field(csn_data),
                        monthly_util_hrs_bbox_json=self._extract_bbox_from_field(monthly_hrs_data),
                        monthly_util_cyc_bbox_json=self._extract_bbox_from_field(monthly_cyc_data),
                        serial_number_bbox_json=self._extract_bbox_from_field(serial_data),
                        serial_number_original_bbox_json=self._extract_bbox_from_field(serial_original_data),
                        location_bbox_json=self._extract_bbox_from_field(location_data),
                        extraction_confidence=comp_data.get('extraction_confidence')
                    )
                    session.add(component)
                    
        except Exception as e:
            logger.warning(f"Could not parse component data: {e}")
    
    def _save_standalone_assets(self, session, extracted_data_id: int, extraction_result: Dict[str, Any]):
        """Save standalone assets data from StandaloneAssetsData model"""
        try:
            fields = extraction_result.get('fields', {})
            
            # Check if we have standalone asset fields
            if any(k in fields for k in ['Month', 'MSN', 'ComponentSerialNumber', 'FlightRegistrationNumber']):
                standalone_asset = StandaloneAssetTable(
                    extracted_data_id=extracted_data_id,
                    month=self._to_str(self._extract_value_from_field(fields.get('Month'))),
                    msn=self._to_str(self._extract_value_from_field(fields.get('MSN'))),
                    flight_registration_number=self._to_str(self._extract_value_from_field(fields.get('FlightRegistrationNumber'))),
                    component_serial_number=self._to_str(self._extract_value_from_field(fields.get('ComponentSerialNumber', '')))
                )
                session.add(standalone_asset)
                
        except Exception as e:
            logger.warning(f"Could not parse standalone assets data: {e}")
    
    def _save_flight_info(self, session, extracted_data_id: int, extraction_result: Dict[str, Any]):
        """Save flight info from FlightInfo model"""
        try:
            fields = extraction_result.get('fields', {})
            
            # Check if we have flight info fields
            if any(k in fields for k in ['Month', 'MSN', 'AirCraftType', 'RegistrationNumber']):
                flight_info = FlightInfoTable(
                    extracted_data_id=extracted_data_id,
                    month=self._to_str(self._extract_value_from_field(fields.get('Month'))),
                    msn=self._to_str(self._extract_value_from_field(fields.get('MSN'))),
                    aircraft_type=self._to_str(self._extract_value_from_field(fields.get('AirCraftType'))),
                    registration_number=self._to_str(self._extract_value_from_field(fields.get('RegistrationNumber')))
                )
                session.add(flight_info)
                
        except Exception as e:
            logger.warning(f"Could not parse flight info data: {e}")
    
    def _to_float(self, value: Any) -> Optional[float]:
        """Convert to float safely"""
        if value is None:
            return None
        try:
            return float(str(value).replace(',', ''))
        except (ValueError, AttributeError):
            return None
    
    def _to_int(self, value: Any) -> Optional[int]:
        """Convert to int safely"""
        if value is None:
            return None
        try:
            return int(float(str(value).replace(',', '')))
        except (ValueError, AttributeError):
            return None
    
    def _to_str(self, value: Any) -> Optional[str]:
        """Convert to string safely"""
        if value is None:
            return None
        return str(value)
    
    def get_all_identifiers(self) -> List[Dict[str, Any]]:
        """Get all identifiers from database"""
        with self.get_session() as session:
            results = session.query(
                Identifier.id,
                Identifier.identifier,
                Identifier.identifier_type,
                Identifier.confidence,
                Document.filename,
                OCRResult.page_number
            ).join(
                OCRResult, Identifier.ocr_result_id == OCRResult.id
            ).join(
                Document, OCRResult.document_id == Document.id
            ).order_by(
                Document.filename,
                OCRResult.page_number
            ).all()
            
            return [
                {
                    'id': row.id,
                    'identifier': row.identifier,
                    'identifier_type': row.identifier_type,
                    'confidence': row.confidence,
                    'filename': row.filename,
                    'page_number': row.page_number
                }
                for row in results
            ]
    
 

    def search_by_identifier(self, identifier: str) -> List[Dict[str, Any]]:
        """
        Search for extraction results by identifier
        Searches in:
        1. Identifiers table (MSN, registration numbers, etc.)
        2. Component serial numbers in components table
        
        Returns unique results with full extraction_json containing bounding boxes
        """
        with self.get_session() as session:
            results = []
            
            # PART 1: Search in Identifiers table (MSN, registration, etc.)
            try:
                identifier_results = session.query(
                    Document.filename,
                    OCRResult.page_number,
                    OCRResult.ocr_json,
                    Identifier.identifier,
                    Identifier.identifier_type,
                    ExtractedData.document_type,
                    ExtractedData.extraction_json,
                    ExtractedData.id.label('extracted_data_id')
                ).join(
                    ExtractedData, ExtractedData.identifier_id == Identifier.id
                ).join(
                    OCRResult, Identifier.ocr_result_id == OCRResult.id
                ).join(
                    Document, OCRResult.document_id == Document.id
                ).filter(
                    Identifier.identifier.like(f'%{identifier}%')
                ).distinct().all()
                
                for row in identifier_results:
                    results.append({
                        'filename': row.filename,
                        'page_number': row.page_number,
                        'identifier': row.identifier,
                        'identifier_type': row.identifier_type,
                        'document_type': row.document_type,
                        'extraction_json': json.loads(row.extraction_json),
                        'ocr_json': json.loads(row.ocr_json),
                        'extracted_data_id': row.extracted_data_id,
                        'search_source': 'identifier'
                    })
            except Exception as e:
                logger.warning(f"Error searching identifiers: {e}")
            
            # PART 2: Search in component serial numbers
            try:
                component_results = session.query(
                    Document.filename,
                    OCRResult.page_number,
                    OCRResult.ocr_json,
                    ComponentTable.serial_number,
                    ComponentTable.component_type,
                    ExtractedData.document_type,
                    ExtractedData.extraction_json,
                    ExtractedData.id.label('extracted_data_id')
                ).join(
                    ExtractedData, ComponentTable.extracted_data_id == ExtractedData.id
                ).join(
                    Identifier, ExtractedData.identifier_id == Identifier.id
                ).join(
                    OCRResult, Identifier.ocr_result_id == OCRResult.id
                ).join(
                    Document, OCRResult.document_id == Document.id
                ).filter(
                    ComponentTable.serial_number.like(f'%{identifier}%')
                ).distinct().all()
                
                for row in component_results:
                    # Get the full extraction_json and filter to just this component
                    full_extraction = json.loads(row.extraction_json)
                    
                    # Create a filtered extraction with only the matching component
                    # This ensures we only highlight the component that matches
                    filtered_extraction = {
                        'document_type': full_extraction.get('document_type'),
                        'layout_type': full_extraction.get('layout_type'),
                        'fields': {
                            row.component_type: full_extraction.get('fields', {}).get(row.component_type, {})
                        }
                    }
                    
                    results.append({
                        'filename': row.filename,
                        'page_number': row.page_number,
                        'identifier': row.serial_number,
                        'identifier_type': 'component_sn',
                        'document_type': row.document_type,
                        'extraction_json': filtered_extraction,  # Only the matching component
                        'ocr_json': json.loads(row.ocr_json),
                        'extracted_data_id': row.extracted_data_id,
                        'search_source': 'component',
                        'component_type': row.component_type
                    })
            except Exception as e:
                logger.warning(f"Error searching components: {e}")
            
            # Remove duplicates based on extracted_data_id and component_type
            seen = set()
            unique_results = []
            for result in results:
                # Create unique key based on extracted_data_id and component (if exists)
                key = (
                    result['extracted_data_id'], 
                    result.get('component_type', ''),
                    result.get('search_source', '')
                )
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            
            logger.info(f"   Found {len(unique_results)} unique result(s) for identifier '{identifier}'")
            
            return unique_results


    def search_by_identifier_with_full_components(self, identifier: str) -> List[Dict[str, Any]]:
        """
        Alternative search that returns ALL components for the matching identifier
        (Not filtered to just the matching component)
        """
        with self.get_session() as session:
            # Search in Identifiers table
            results = session.query(
                Document.filename,
                OCRResult.page_number,
                OCRResult.ocr_json,
                Identifier.identifier,
                Identifier.identifier_type,
                ExtractedData.document_type,
                ExtractedData.extraction_json
            ).join(
                ExtractedData, ExtractedData.identifier_id == Identifier.id
            ).join(
                OCRResult, Identifier.ocr_result_id == OCRResult.id
            ).join(
                Document, OCRResult.document_id == Document.id
            ).filter(
                Identifier.identifier.like(f'%{identifier}%')
            ).distinct().all()
            
            return [
                {
                    'filename': row.filename,
                    'page_number': row.page_number,
                    'identifier': row.identifier,
                    'identifier_type': row.identifier_type,
                    'document_type': row.document_type,
                    'extraction_json': json.loads(row.extraction_json),
                    'ocr_json': json.loads(row.ocr_json)
                }
                for row in results
            ]
            
    def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get document info by filename"""
        with self.get_session() as session:
            try:
                doc = session.query(Document).filter(
                    Document.filename == filename
                ).first()
                
                if doc:
                    return {
                        'id': doc.id,
                        'filename': doc.filename,
                        'file_path': doc.file_path,
                        'total_pages': doc.total_pages,
                        'processed_date': doc.processed_date
                    }
                return None
                
            except Exception as e:
                logger.error(f"Error getting document: {e}")
                return None
    
    def get_ocr_by_document_and_page(self, document_id: int, page_number: int) -> Optional[Dict[str, Any]]:
        """Get OCR result for specific document and page"""
        with self.get_session() as session:
            try:
                ocr = session.query(OCRResult).filter(
                    OCRResult.document_id == document_id,
                    OCRResult.page_number == page_number
                ).first()
                
                if ocr:
                    return {
                        'id': ocr.id,
                        'document_id': ocr.document_id,
                        'page_number': ocr.page_number,
                        'total_blocks': ocr.total_blocks,
                        'image_width': ocr.image_width,
                        'image_height': ocr.image_height,
                        'ocr_json': ocr.ocr_json,
                        'processed_date': ocr.processed_date
                    }
                return None
                
            except Exception as e:
                logger.error(f"Error getting OCR result: {e}")
                return None
    
    def get_components_by_identifier(self, identifier: str) -> List[Dict[str, Any]]:
        """Get component data for a specific identifier"""
        with self.get_session() as session:
            results = session.query(ComponentTable).join(
                ExtractedData, ComponentTable.extracted_data_id == ExtractedData.id
            ).join(
                Identifier, ExtractedData.identifier_id == Identifier.id
            ).filter(
                Identifier.identifier.like(f'%{identifier}%')
            ).all()
            
            return [
                {
                    'component_type': comp.component_type,
                    'tsn': comp.tsn,
                    'csn': comp.csn,
                    'monthly_util_hrs': comp.monthly_util_hrs,
                    'monthly_util_cyc': comp.monthly_util_cyc,
                    'serial_number': comp.serial_number,
                    'location': comp.location,
                    'derate': comp.derate,
                    'extraction_confidence': comp.extraction_confidence
                }
                for comp in results
            ]
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()