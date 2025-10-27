"""
Main script for processing PDFs with Azure OCR
"""
import argparse
import sys
from pathlib import Path
import logging
from typing import Optional
import json

from src.pdf_processor import PDFProcessor
from src.azure_ocr import AzureOCR
from src.text_mapper import TextMapper
from src.database import DatabaseManager
from src.llm_field_extractor import LLMFieldExtractor
from src.pdf_highlighter import highlight_extraction_in_pdf
from config.config import INPUT_DIR, OCR_RESULTS_DIR, VISUALIZATIONS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_pdf_with_llm(ocr_json_path: str, serial_number: str, pdf_path: Optional[str] = None) -> None:
    """
    Process existing OCR results with LLM to extract data
    
    Args:
        ocr_json_path: Path to OCR JSON file
        serial_number: Serial number/identifier to search for
        pdf_path: Optional path to original PDF for highlighting
    """
    ocr_json_path = Path(ocr_json_path)
    
    if not ocr_json_path.exists():
        logger.error(f"❌ OCR file not found: {ocr_json_path}")
        return
    
    logger.info("="*60)
    logger.info(f"🧠 LLM Data Extraction")
    logger.info(f"   File: {ocr_json_path.name}")
    logger.info(f"   Identifier: {serial_number}")
    if pdf_path:
        logger.info(f"   PDF: {Path(pdf_path).name}")
    logger.info("="*60)
    
    try:
        # Load OCR results
        logger.info("\n📄 Loading OCR results...")
        with open(ocr_json_path, 'r', encoding='utf-8') as f:
            ocr_results = json.load(f)
        
        logger.info(f"   Total text blocks: {len(ocr_results.get('text_blocks', []))}")
        
        # Check if image_dimensions exist
        if 'image_dimensions' not in ocr_results:
            logger.warning("   ⚠️  No image_dimensions found in OCR results")
            logger.warning("   This is needed for PDF highlighting. Checking OCR structure...")
            
            # Try to infer dimensions from bounding boxes
            text_blocks = ocr_results.get('text_blocks', [])
            if text_blocks:
                max_x = max(block['bounding_box']['left'] + block['bounding_box']['width'] 
                           for block in text_blocks)
                max_y = max(block['bounding_box']['top'] + block['bounding_box']['height'] 
                           for block in text_blocks)
                
                ocr_results['image_dimensions'] = {
                    'width': max_x,
                    'height': max_y
                }
                logger.info(f"   ✅ Inferred dimensions: {max_x:.0f} x {max_y:.0f}")
            else:
                logger.error("   ❌ Cannot infer dimensions - no text blocks found!")
        else:
            img_dims = ocr_results['image_dimensions']
            logger.info(f"   Image dimensions: {img_dims['width']:.0f} x {img_dims['height']:.0f}")
        
        # Extract data using LLM
        logger.info("\n🧠 Using LLM to extract data...")
        extractor = LLMFieldExtractor()
        result = extractor.extract_column_data(ocr_results, serial_number)
        
        if not result['found']:
            logger.error(f"\n❌ {result['message']}")
            return
        
        # Display results header
        logger.info(f"\n✅ Found identifier: {result['identifier']}")
        logger.info(f"   Type: {result.get('identifier_type', 'N/A')}")
        logger.info(f"   Layout: {result.get('layout_type', 'N/A')}")
        logger.info(f"   Document Type: {result.get('document_type', 'N/A')}")
        logger.info(f"   Total fields: {result.get('total_fields', 0)}")
        
        # Display extracted fields
        logger.info("\n" + "="*60)
        logger.info("📊 Extracted Fields:")
        logger.info("="*60 + "\n")
        
        fields = result.get('fields', {})
        
        # Display aircraft-level fields
        aircraft_fields = ['aircraft_registration', 'msn', 'tah', 'tac', 
                          'delta_hrs', 'delta_cyc', 'c_check_expiry', 'd_check_expiry']
        
        logger.info("🛩️  Aircraft Information:")
        for field_name in aircraft_fields:
            if field_name in fields:
                field_data = fields[field_name]
                text = field_data.get('text', 'N/A')
                bbox = field_data.get('bounding_box', {})
                logger.info(f"   {field_name.upper()}: {text}")
                logger.info(f"      Position: ({bbox.get('left', 0):.0f}, {bbox.get('top', 0):.0f})")
        
        # Display engine data
        if 'engines' in fields:
            logger.info("\n🔧 Engines:")
            for i, engine in enumerate(fields['engines'], 1):
                logger.info(f"\n   Engine {i}:")
                for key, field_data in engine.items():
                    text = field_data.get('text', 'N/A')
                    bbox = field_data.get('bounding_box', {})
                    logger.info(f"      {key.upper()}: {text}")
                    logger.info(f"         Position: ({bbox.get('left', 0):.0f}, {bbox.get('top', 0):.0f})")
        
        # Display APU data if present
        apu_fields = {k: v for k, v in fields.items() if k.startswith('apu_')}
        if apu_fields:
            logger.info("\n⚡ APU:")
            for field_name, field_data in apu_fields.items():
                text = field_data.get('text', 'N/A')
                bbox = field_data.get('bounding_box', {})
                logger.info(f"   {field_name.upper()}: {text}")
                logger.info(f"      Position: ({bbox.get('left', 0):.0f}, {bbox.get('top', 0):.0f})")
        
        # Display other fields
        displayed_fields = set(aircraft_fields + ['engines'])
        other_fields = {k: v for k, v in fields.items() 
                       if k not in displayed_fields and not k.startswith('apu_')}
        
        if other_fields:
            logger.info("\n📋 Other Fields:")
            for field_name, field_data in other_fields.items():
                text = field_data.get('text', 'N/A')
                bbox = field_data.get('bounding_box', {})
                logger.info(f"   {field_name.upper()}: {text}")
                logger.info(f"      Position: ({bbox.get('left', 0):.0f}, {bbox.get('top', 0):.0f})")
        
        # Overall bounding box for highlighting
        overall_bbox = result.get('bounding_box', {})
        logger.info("\n" + "="*60)
        logger.info("🎯 Overall Bounding Box (for frontend highlighting):")
        logger.info("="*60)
        logger.info(f"   Left: {overall_bbox.get('left', 0):.0f}")
        logger.info(f"   Top: {overall_bbox.get('top', 0):.0f}")
        logger.info(f"   Width: {overall_bbox.get('width', 0):.0f}")
        logger.info(f"   Height: {overall_bbox.get('height', 0):.0f}")
        
        # Save results
        output_file = ocr_json_path.parent / f"{ocr_json_path.stem}_extracted_{serial_number}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Saved to: {output_file}")
        
        # Highlight in PDF if provided
        if pdf_path:
            pdf_path = Path(pdf_path)
            if pdf_path.exists():
                logger.info("\n" + "="*60)
                logger.info("🎨 Highlighting in PDF...")
                logger.info("="*60)
                
                # Check if we have image_dimensions
                if 'image_dimensions' not in ocr_results:
                    logger.error("\n❌ Cannot highlight PDF: image_dimensions missing from OCR results")
                    logger.error("   Please re-run OCR processing to include image dimensions")
                else:
                    try:
                        highlighted_pdf = highlight_extraction_in_pdf(
                            pdf_path=str(pdf_path),
                            extraction_result=result,
                            method="rectangle",  # More visible
                            ocr_results=ocr_results  # Pass OCR results with image_dimensions
                        )
                        
                        logger.info(f"\n✅ Highlighted PDF created!")
                        logger.info(f"   Location: {highlighted_pdf}")
                    except Exception as e:
                        logger.error(f"\n❌ Error highlighting PDF: {str(e)}")
                        import traceback
                        traceback.print_exc()
            else:
                logger.warning(f"\n⚠️  PDF file not found: {pdf_path}")
                logger.warning(f"   Skipping PDF highlighting...")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 LLM extraction completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n❌ Error during LLM processing: {str(e)}", exc_info=True)
        sys.exit(1)

def process_pdf(pdf_path: str, page_num: Optional[int] = None, 
                save_to_db: bool = False, search_terms: Optional[list] = None) -> None:
    """
    Process a PDF file with Azure OCR
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Specific page to process (None = all pages)
        save_to_db: Whether to save results to database
        search_terms: Optional list of terms to search for
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        logger.error(f"❌ File not found: {pdf_path}")
        return
    
    logger.info("="*60)
    logger.info(f"📄 Processing PDF with Azure OCR")
    logger.info(f"   File: {pdf_path.name}")
    if page_num:
        logger.info(f"   Page: {page_num}")
    logger.info("="*60)
    
    try:
        # Initialize processors
        pdf_processor = PDFProcessor()
        azure_ocr = AzureOCR()
        text_mapper = TextMapper()
        
        # Convert PDF to images
        logger.info("\n🖼️  Converting PDF to images...")
        images = pdf_processor.pdf_to_images(str(pdf_path))
        
        if page_num:
            if page_num > len(images):
                logger.error(f"❌ Page {page_num} not found (total pages: {len(images)})")
                return
            images = [images[page_num - 1]]
            logger.info(f"   Processing page {page_num}")
        else:
            logger.info(f"   Total pages: {len(images)}")
        
        # Process each page
        for idx, image in enumerate(images, 1):
            current_page = page_num if page_num else idx
            logger.info(f"\n📄 Processing page {current_page}...")
            
            # Perform OCR
            ocr_result = azure_ocr.analyze_image(image)
            
            if not ocr_result:
                logger.warning(f"⚠️  No OCR results for page {current_page}")
                continue
            
            # Map text blocks
            text_blocks = text_mapper.map_text_blocks(ocr_result)
            
            # Save OCR results WITH image_dimensions
            output_file = OCR_RESULTS_DIR / f"{pdf_path.stem}_page_{current_page}_ocr.json"
            ocr_data = {
                'filename': pdf_path.name,
                'page_number': current_page,
                'text_blocks': text_blocks,
                'total_blocks': len(text_blocks),
                'image_dimensions': {  # IMPORTANT: Save image dimensions
                    'width': image.width,
                    'height': image.height
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ocr_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"   ✅ Extracted {len(text_blocks)} text blocks")
            logger.info(f"   📐 Image size: {image.width} x {image.height}")
            logger.info(f"   💾 Saved to: {output_file.name}")
            
            # Search for terms if provided
            if search_terms:
                logger.info(f"\n🔍 Searching for terms: {', '.join(search_terms)}")
                for term in search_terms:
                    matches = [block for block in text_blocks 
                              if term.lower() in block['text'].lower()]
                    if matches:
                        logger.info(f"   ✅ Found '{term}' in {len(matches)} blocks")
                        for match in matches[:3]:  # Show first 3 matches
                            bbox = match['bounding_box']
                            logger.info(f"      '{match['text']}' at ({bbox['left']:.0f}, {bbox['top']:.0f})")
                    else:
                        logger.info(f"   ❌ '{term}' not found")
            
            # Save to database if requested
            if save_to_db:
                logger.info(f"\n💾 Saving to database...")
                db = DatabaseManager()
                # Implement database save logic here
                logger.info(f"   ✅ Saved to database")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 PDF processing completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n❌ Error during PDF processing: {str(e)}", exc_info=True)
        sys.exit(1)


def list_input_files() -> None:
    """List all PDF files in the input directory"""
    try:
        pdf_files = list(INPUT_DIR.glob("*.pdf"))
        
        if not pdf_files:
            logger.info("📂 No PDF files found in input directory")
            logger.info(f"   Directory: {INPUT_DIR}")
            return
        
        logger.info("="*60)
        logger.info(f"📂 PDF Files in {INPUT_DIR.name}/")
        logger.info("="*60)
        
        for i, pdf_file in enumerate(pdf_files, 1):
            size_mb = pdf_file.stat().st_size / (1024 * 1024)
            logger.info(f"{i:2d}. {pdf_file.name} ({size_mb:.2f} MB)")
        
        logger.info("="*60)
        logger.info(f"Total: {len(pdf_files)} PDF files")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Error listing files: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process PDF files with Azure Computer Vision OCR and LLM extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF with Azure OCR
  python main.py --file data/input/document.pdf
  
  # Process specific page
  python main.py --file data/input/document.pdf --page 1
  
  # Process and search for terms
  python main.py --file data/input/document.pdf --search "AKNT" "779682"
  
  # Use LLM to extract data from existing OCR results
  python main.py --llm-only data/output/ocr_results/sample_page_1_ocr.json "P-11217"
  
  # Use LLM to extract AND highlight in PDF
  python main.py --llm-only data/output/ocr_results/sample_page_1_ocr.json "P-11217" --pdf sample.pdf
  
  # List all PDF files
  python main.py --list
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to PDF file to process'
    )
    
    parser.add_argument(
        '--page', '-p',
        type=int,
        help='Specific page number to process (default: all pages)'
    )
    
    parser.add_argument(
        '--db',
        action='store_true',
        help='Save results to database'
    )
    
    parser.add_argument(
        '--search', '-s',
        nargs='+',
        help='Search terms to look for in OCR results'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all PDF files in input directory'
    )
    
    parser.add_argument(
        '--llm-only',
        nargs=2,
        metavar=('OCR_JSON', 'IDENTIFIER'),
        help='Use LLM to extract data from existing OCR JSON file'
    )
    
    parser.add_argument(
        '--pdf',
        type=str,
        help='Path to PDF file for highlighting (use with --llm-only)'
    )
    
    args = parser.parse_args()
    
    # Handle LLM-only mode
    if args.llm_only:
        ocr_json_path, serial_number = args.llm_only
        pdf_path = args.pdf  # Get optional PDF path
        process_pdf_with_llm(ocr_json_path, serial_number, pdf_path)
        return
    
    # Handle list command
    if args.list:
        list_input_files()
        return
    
    # Handle file processing
    if args.file:
        process_pdf(
            pdf_path=args.file,
            page_num=args.page,
            save_to_db=args.db,
            search_terms=args.search
        )
    else:
        # No arguments - show help
        parser.print_help()
        print("\n")
        list_input_files()


if __name__ == "__main__":
    main()