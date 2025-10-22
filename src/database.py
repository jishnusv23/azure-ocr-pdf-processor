"""
Database operations for storing OCR results
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import json

from config.config import DATABASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class PDFDocument(Base):
    """PDF Document model"""
    __tablename__ = 'pdf_documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    total_pages = Column(Integer)
    file_size_mb = Column(Float)
    processed_at = Column(DateTime, default=datetime.utcnow)
    pdf_metadata = Column(JSON)  # ✅ CHANGED: from 'metadata' to 'pdf_metadata'


class OCRResult(Base):
    """OCR Result model"""
    __tablename__ = 'ocr_results'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    image_width = Column(Integer)
    image_height = Column(Integer)
    total_text_blocks = Column(Integer)
    all_text = Column(Text)
    ocr_data = Column(JSON)
    processed_at = Column(DateTime, default=datetime.utcnow)


class TextBlock(Base):
    """Individual text block model"""
    __tablename__ = 'text_blocks'
    
    id = Column(Integer, primary_key=True)
    ocr_result_id = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    confidence = Column(Float)
    left = Column(Float)
    top = Column(Float)
    right = Column(Float)
    bottom = Column(Float)
    width = Column(Float)
    height = Column(Float)


class DatabaseManager:
    """Manages database operations"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database connection
        
        Args:
            database_url: Database connection URL (uses config if not provided)
        """
        self.database_url = database_url or DATABASE_URL
        
        if not self.database_url:
            logger.warning("⚠️  Database URL not configured. Database operations disabled.")
            self.engine = None
            self.Session = None
            return
        
        try:
            self.engine = create_engine(self.database_url, echo=False)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {str(e)}")
            self.engine = None
            self.Session = None
    
    def create_tables(self):
        """Create all tables in the database"""
        if not self.engine:
            logger.error("❌ Database not configured")
            return
        
        try:
            Base.metadata.create_all(self.engine)
            logger.info("✅ Database tables created successfully")
        except Exception as e:
            logger.error(f"❌ Error creating tables: {str(e)}")
            raise
    
    def save_document(self, filename: str, file_path: str, total_pages: int, 
                     file_size_mb: float, pdf_metadata: Dict[str, Any] = None) -> int:  # ✅ CHANGED parameter name
        """
        Save PDF document information
        
        Returns:
            Document ID
        """
        if not self.Session:
            logger.error("❌ Database not configured")
            return -1
        
        session = self.Session()
        
        try:
            document = PDFDocument(
                filename=filename,
                file_path=file_path,
                total_pages=total_pages,
                file_size_mb=file_size_mb,
                pdf_metadata=pdf_metadata or {}  # ✅ CHANGED: use 'pdf_metadata'
            )
            
            session.add(document)
            session.commit()
            doc_id = document.id
            
            logger.info(f"💾 Saved document: {filename} (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving document: {str(e)}")
            raise
        finally:
            session.close()
    
    def save_ocr_result(self, document_id: int, page_number: int, 
                       ocr_data: Dict[str, Any]) -> int:
        """
        Save OCR results for a page
        
        Returns:
            OCR result ID
        """
        if not self.Session:
            logger.error("❌ Database not configured")
            return -1
        
        session = self.Session()
        
        try:
            image_size = ocr_data.get('image_size', {})
            
            ocr_result = OCRResult(
                document_id=document_id,
                page_number=page_number,
                image_width=image_size.get('width'),
                image_height=image_size.get('height'),
                total_text_blocks=len(ocr_data.get('text_blocks', [])),
                all_text=ocr_data.get('all_text', ''),
                ocr_data=ocr_data
            )
            
            session.add(ocr_result)
            session.commit()
            result_id = ocr_result.id
            
            logger.info(f"💾 Saved OCR result for page {page_number} (ID: {result_id})")
            
            # Save individual text blocks
            self._save_text_blocks(session, result_id, ocr_data.get('text_blocks', []))
            
            return result_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving OCR result: {str(e)}")
            raise
        finally:
            session.close()
    
    def _save_text_blocks(self, session, ocr_result_id: int, text_blocks: List[Dict[str, Any]]):
        """Save individual text blocks"""
        try:
            for block in text_blocks:
                bbox = block.get('bounding_box', {})
                
                text_block = TextBlock(
                    ocr_result_id=ocr_result_id,
                    text=block.get('text', ''),
                    confidence=block.get('confidence'),
                    left=bbox.get('left'),
                    top=bbox.get('top'),
                    right=bbox.get('right'),
                    bottom=bbox.get('bottom'),
                    width=bbox.get('width'),
                    height=bbox.get('height')
                )
                
                session.add(text_block)
            
            session.commit()
            logger.info(f"💾 Saved {len(text_blocks)} text blocks")
            
        except Exception as e:
            logger.error(f"❌ Error saving text blocks: {str(e)}")
            raise
    
    def search_text(self, search_term: str, document_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for text in stored OCR results
        
        Args:
            search_term: Text to search for
            document_id: Optional document ID to limit search
            
        Returns:
            List of matching text blocks with positions
        """
        if not self.Session:
            logger.error("❌ Database not configured")
            return []
        
        session = self.Session()
        
        try:
            query = session.query(TextBlock)
            
            if document_id:
                query = query.join(OCRResult).filter(OCRResult.document_id == document_id)
            
            query = query.filter(TextBlock.text.ilike(f'%{search_term}%'))
            
            results = []
            for block in query.all():
                results.append({
                    'text': block.text,
                    'confidence': block.confidence,
                    'bounding_box': {
                        'left': block.left,
                        'top': block.top,
                        'right': block.right,
                        'bottom': block.bottom,
                        'width': block.width,
                        'height': block.height
                    }
                })
            
            logger.info(f"🔍 Found {len(results)} matches for '{search_term}'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching text: {str(e)}")
            return []
        finally:
            session.close()
    
    def get_document_ocr_results(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all OCR results for a document"""
        if not self.Session:
            logger.error("❌ Database not configured")
            return []
        
        session = self.Session()
        
        try:
            results = session.query(OCRResult).filter(
                OCRResult.document_id == document_id
            ).order_by(OCRResult.page_number).all()
            
            return [
                {
                    'page_number': r.page_number,
                    'total_text_blocks': r.total_text_blocks,
                    'ocr_data': r.ocr_data
                }
                for r in results
            ]
            
        finally:
            session.close()


if __name__ == "__main__":
    # Test database operations
    db = DatabaseManager()
    
    if db.engine:
        # Create tables
        db.create_tables()
        
        # Test saving a document
        doc_id = db.save_document(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            total_pages=2,
            file_size_mb=1.5,
            pdf_metadata={"test": "data"}  # ✅ CHANGED parameter name
        )
        
        print(f"✅ Created document with ID: {doc_id}")
    else:
        print("⚠️  Database not configured. Skipping tests.")