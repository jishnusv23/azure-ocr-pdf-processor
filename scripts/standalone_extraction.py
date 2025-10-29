"""
Standalone PDF extraction - COMPLETE MULTI-PAGE SUPPORT
Highlights ALL fields across ALL pages for each identifier
Location: scripts/standalone_extraction.py
"""
import argparse
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_processor import PDFProcessor
from src.azure_ocr import AzureOCR
from src.text_mapper import TextMapper
from src.llm_field_extractor import LLMFieldExtractor
from src.pdf_highlighter import PDFHighlighter
from config.config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_pdf_standalone(pdf_path: str, output_dir: str = None) -> None:
    """
    COMPLETE MULTI-PAGE PROCESSING
    - Processes ALL pages at once
    - Extracts identifiers and ALL their fields across ALL pages
    - Creates ONE highlighted PDF per identifier showing ALL fields on ALL pages
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Output directory for highlighted PDFs
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        logger.error(f"❌ File not found: {pdf_path}")
        return
    
    if not output_dir:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info(f"🚀 MULTI-PAGE EXTRACTION PIPELINE")
    logger.info(f"   Input PDF: {pdf_path.name}")
    logger.info(f"   Output Dir: {output_dir}")
    logger.info("="*80)
    
    try:
        # Initialize processors
        pdf_processor = PDFProcessor()
        azure_ocr = AzureOCR()
        text_mapper = TextMapper()
        llm_extractor = LLMFieldExtractor()
        
        # Convert PDF to images
        logger.info("\n🖼️  Step 1: Converting PDF to images...")
        images = pdf_processor.pdf_to_images(str(pdf_path))
        logger.info(f"   ✅ Total pages: {len(images)}")
        
        # Step 2: Process ALL pages and collect ALL OCR data
        logger.info("\n📄 Step 2: Processing ALL pages (OCR)...")
        all_ocr_data = []  # Store OCR for each page
        
        for page_idx, image in enumerate(images, 1):
            logger.info(f"   Page {page_idx}: Performing Azure OCR...")
            ocr_result = azure_ocr.analyze_image(image)
            
            if not ocr_result:
                logger.warning(f"   ⚠️  No OCR results for page {page_idx}")
                continue
            
            text_blocks = text_mapper.map_text_blocks(ocr_result)
            
            ocr_data = {
                'page_number': page_idx,
                'text_blocks': text_blocks,
                'image_dimensions': {
                    'width': image.width,
                    'height': image.height
                }
            }
            
            all_ocr_data.append(ocr_data)
            logger.info(f"   ✅ Page {page_idx}: {len(text_blocks)} blocks extracted")
        
        if not all_ocr_data:
            logger.error("❌ No OCR data extracted from any page")
            return
        
        # Step 3: LLM Extraction - Process ALL pages together
        logger.info(f"\n{'='*80}")
        logger.info(f"🧠 Step 3: LLM Extraction (ALL {len(all_ocr_data)} pages together)")
        logger.info(f"{'='*80}")
        
        # Combine all OCR data with page numbers
        combined_extraction = llm_extractor.extract_all_data_multipage(
            all_ocr_data=all_ocr_data,
            pdf_filename=pdf_path.name
        )
        
        if not combined_extraction:
            logger.warning("⚠️  No identifiers found in entire PDF")
            return
        
        logger.info(f"   ✅ Found {len(combined_extraction)} identifiers across all pages")
        
        # Step 4: Generate highlighted PDFs
        logger.info(f"\n{'='*80}")
        logger.info(f"🎨 Step 4: Generating Multi-Page Highlighted PDFs")
        logger.info(f"{'='*80}")
        
        generated_pdfs = []
        
        for idx, identifier_data in enumerate(combined_extraction, 1):
            identifier = identifier_data['identifier']
            fields_by_page = identifier_data['fields_by_page']
            
            logger.info(f"\n{idx}. Highlighting: {identifier}")
            logger.info(f"   Pages with data: {list(fields_by_page.keys())}")
            
            # Generate output filename
            safe_identifier = identifier.replace('/', '-').replace('\\', '-')
            output_filename = f"{pdf_path.stem}_{safe_identifier}_multipage_highlighted.pdf"
            output_path = output_dir / output_filename
            
            try:
                # Create multi-page highlighted PDF
                highlighter = PDFHighlighter(opacity=0.4)
                highlighted_pdf = highlighter.highlight_multipage(
                    pdf_path=str(pdf_path),
                    fields_by_page=fields_by_page,
                    output_path=str(output_path),
                    all_ocr_data=all_ocr_data
                )
                
                total_fields = sum(len(fields) for fields in fields_by_page.values())
                
                generated_pdfs.append({
                    'identifier': identifier,
                    'pages': list(fields_by_page.keys()),
                    'file': highlighted_pdf,
                    'total_fields': total_fields
                })
                
                logger.info(f"   ✅ Created: {output_filename}")
                logger.info(f"   📊 Total fields highlighted: {total_fields}")
                logger.info(f"   📄 Pages: {list(fields_by_page.keys())}")
            
            except Exception as e:
                logger.error(f"   ❌ Error highlighting {identifier}: {str(e)}\n")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("🎉 MULTI-PAGE EXTRACTION COMPLETED!")
        logger.info("="*80)
        logger.info(f"   📄 Input PDF: {pdf_path.name}")
        logger.info(f"   📄 Pages processed: {len(images)}")
        logger.info(f"   🔍 Identifiers found: {len(combined_extraction)}")
        logger.info(f"   ✅ PDFs generated: {len(generated_pdfs)}")
        logger.info(f"\n   📁 Output Directory: {output_dir}")
        logger.info(f"\n   📋 Generated Files:")
        
        for i, pdf_info in enumerate(generated_pdfs, 1):
            logger.info(f"      {i}. {Path(pdf_info['file']).name}")
            logger.info(f"         → {pdf_info['identifier']}")
            logger.info(f"         → Pages: {pdf_info['pages']}")
            logger.info(f"         → Fields: {pdf_info['total_fields']}")
        
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n❌ Error during pipeline: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Page PDF Extraction - Highlights ALL fields across ALL pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python standalone_extraction.py --file data/sample.pdf
  python standalone_extraction.py --file data/sample3.pdf --output-dir results/

Features:
  - Processes ALL pages in one operation
  - Creates ONE PDF per identifier
  - Highlights ALL fields on ALL pages where they appear
  - Handles 2-page, 3-page, or any multi-page PDFs
        """
    )
    
    parser.add_argument('--file', '-f', type=str, required=True,
                       help='Path to PDF file to process')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory (default: data/output)')
    
    args = parser.parse_args()
    process_pdf_standalone(pdf_path=args.file, output_dir=args.output_dir)


if __name__ == "__main__":
    main()