"""
PDF Processor - Converts PDF pages to images for OCR processing
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple
import logging
from PIL import Image
import io

from config.config import PDF_DPI, IMAGE_FORMAT, CACHE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF to image conversion"""
    
    def __init__(self, dpi: int = PDF_DPI):
        self.dpi = dpi
        self.zoom = dpi / 72  # PDF default DPI is 72
        
    def convert_pdf_to_images(self, pdf_path: str) -> List[Tuple[int, Image.Image]]:
        """
        Convert PDF pages to PIL Images
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of tuples (page_number, PIL.Image)
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"📄 Processing PDF: {pdf_path.name}")
        logger.info(f"   DPI: {self.dpi}")
        
        images = []
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(str(pdf_path))
            total_pages = len(pdf_document)
            
            logger.info(f"   Total pages: {total_pages}")
            
            # Process each page
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                
                # Convert page to image with specified DPI
                mat = fitz.Matrix(self.zoom, self.zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to PIL Image
                img_data = pix.tobytes(IMAGE_FORMAT.lower())
                img = Image.open(io.BytesIO(img_data))
                
                images.append((page_num + 1, img))
                logger.info(f"   ✅ Page {page_num + 1} converted ({img.size[0]}x{img.size[1]})")
            
            pdf_document.close()
            
            logger.info(f"✅ Successfully converted {total_pages} pages")
            return images
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF: {str(e)}")
            raise
    
    def save_image(self, image: Image.Image, output_path: str) -> None:
        """Save PIL Image to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        image.save(str(output_path), format=IMAGE_FORMAT)
        logger.info(f"💾 Saved image: {output_path}")
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """Get PDF metadata"""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        pdf_document = fitz.open(str(pdf_path))
        
        info = {
            "filename": pdf_path.name,
            "total_pages": len(pdf_document),
            "metadata": pdf_document.metadata,
            "file_size_mb": pdf_path.stat().st_size / (1024 * 1024)
        }
        
        pdf_document.close()
        return info


if __name__ == "__main__":
    # Test the PDF processor
    processor = PDFProcessor()
    
    # Example usage
    test_pdf = CACHE_DIR.parent / "input" / "test.pdf"
    
    if test_pdf.exists():
        # Get PDF info
        info = processor.get_pdf_info(str(test_pdf))
        print(f"\n📊 PDF Info:")
        print(f"   Filename: {info['filename']}")
        print(f"   Pages: {info['total_pages']}")
        print(f"   Size: {info['file_size_mb']:.2f} MB")
        
        # Convert to images
        images = processor.convert_pdf_to_images(str(test_pdf))
        print(f"\n✅ Converted {len(images)} pages to images")
        
        # Save first page as example
        if images:
            output_path = CACHE_DIR / f"{test_pdf.stem}_page_1.png"
            processor.save_image(images[0][1], str(output_path))
    else:
        print(f"⚠️  Test PDF not found: {test_pdf}")
        print(f"   Place a PDF file in: {test_pdf.parent}")