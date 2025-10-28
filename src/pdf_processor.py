"""
PDF Processor for converting PDFs to images
"""
import logging
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF to image conversion"""
    
    def __init__(self, dpi: int = 300):
        """
        Initialize PDF Processor
        
        Args:
            dpi: Resolution for image conversion (default: 300)
        """
        self.dpi = dpi
        self.zoom = dpi / 72  # 72 is default PDF DPI
        logger.info(f"✅ PDF Processor initialized (DPI: {dpi})")
    
    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Image objects
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PIL Image objects (one per page)
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"📄 Converting PDF: {pdf_path.name}")
        
        try:
            # Open PDF
            pdf_document = fitz.open(str(pdf_path))
            total_pages = len(pdf_document)
            logger.info(f"   Total pages: {total_pages}")
            
            images = []
            
            # Convert each page to image
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                
                # Render page to pixmap (image)
                mat = fitz.Matrix(self.zoom, self.zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert pixmap to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                images.append(img)
                logger.info(f"   ✅ Page {page_num + 1}: {pix.width}x{pix.height}px")
            
            pdf_document.close()
            
            logger.info(f"✅ Converted {len(images)} pages successfully")
            return images
            
        except Exception as e:
            logger.error(f"❌ Error converting PDF: {str(e)}")
            raise
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[Tuple[int, Image.Image]]:
        """
        Convert PDF to list of (page_number, PIL Image) tuples
        (Alternative method that includes page numbers)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of tuples: (page_number, PIL_Image)
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"📄 Converting PDF: {pdf_path.name}")
        
        try:
            # Open PDF
            pdf_document = fitz.open(str(pdf_path))
            total_pages = len(pdf_document)
            logger.info(f"   Total pages: {total_pages}")
            
            images = []
            
            # Convert each page to image
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                
                # Render page to pixmap (image)
                mat = fitz.Matrix(self.zoom, self.zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert pixmap to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                images.append((page_num + 1, img))  # Page numbers start from 1
                logger.info(f"   ✅ Page {page_num + 1}: {pix.width}x{pix.height}px")
            
            pdf_document.close()
            
            logger.info(f"✅ Converted {len(images)} pages successfully")
            return images
            
        except Exception as e:
            logger.error(f"❌ Error converting PDF: {str(e)}")
            raise
    
    def get_pdf_metadata(self, pdf_path: str) -> dict:
        """
        Extract PDF metadata
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            pdf_document = fitz.open(str(pdf_path))
            
            metadata = {
                'title': pdf_document.metadata.get('title', ''),
                'author': pdf_document.metadata.get('author', ''),
                'subject': pdf_document.metadata.get('subject', ''),
                'keywords': pdf_document.metadata.get('keywords', ''),
                'creator': pdf_document.metadata.get('creator', ''),
                'producer': pdf_document.metadata.get('producer', ''),
                'creation_date': pdf_document.metadata.get('creationDate', ''),
                'modification_date': pdf_document.metadata.get('modDate', ''),
                'total_pages': len(pdf_document)
            }
            
            pdf_document.close()
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Error reading PDF metadata: {str(e)}")
            return {}


if __name__ == "__main__":
    # Test PDF processor
    import sys
    
    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]
    else:
        # Try to find a test PDF
        from config.config import INPUT_DIR
        pdf_files = list(INPUT_DIR.glob("*.pdf"))
        
        if pdf_files:
            test_pdf = pdf_files[0]
        else:
            print("❌ No PDF files found for testing")
            sys.exit(1)
    
    print(f"\n🧪 Testing with: {test_pdf}")
    
    processor = PDFProcessor(dpi=300)
    
    # Test metadata extraction
    print("\n📋 PDF Metadata:")
    metadata = processor.get_pdf_metadata(str(test_pdf))
    for key, value in metadata.items():
        if value:
            print(f"   {key}: {value}")
    
    # Test image conversion
    print("\n🖼️  Converting to images:")
    images = processor.pdf_to_images(str(test_pdf))
    
    print(f"\n✅ Successfully converted {len(images)} pages")
    print(f"   First page size: {images[0].size}")