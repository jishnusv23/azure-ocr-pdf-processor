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
from config.config import INPUT_DIR, OCR_RESULTS_DIR, VISUALIZATIONS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_pdf_with_llm(ocr_json_path: str, serial_number: str) -> None:
    """
    Process existing OCR results with LLM to extract column data
    
    Args:
        ocr_json_path: Path to OCR JSON file
        serial_number: Serial number to search for
    """
    ocr_json_path = Path(ocr_json_path)
    
    if not ocr_json_path.exists():
        logger.error(f"❌ OCR file not found: {ocr_json_path}")
        return
    
    logger.info("="*60)
    logger.info(f"🧠 LLM Column Extraction")
    logger.info(f"   File: {ocr_json_path.name}")
    logger.info(f"   Serial Number: {serial_number}")
    logger.info("="*60)
    
    try:
        # Load OCR results
        logger.info("\n📄 Loading OCR results...")
        with open(ocr_json_path, 'r', encoding='utf-8') as f:
            ocr_results = json.load(f)
        
        logger.info(f"   Total text blocks: {len(ocr_results.get('text_blocks', []))}")
        
        # Extract column data using LLM
        logger.info("\n🧠 Using LLM to extract column data...")
        extractor = LLMFieldExtractor()
        result = extractor.extract_column_data(ocr_results, serial_number)
        
        if not result['found']:
            logger.error(f"\n❌ {result['message']}")
            return
        
        # Display results
        logger.info(f"\n✅ Found serial number: {result['serial_number']}")
        logger.info(f"   Total items in column: {result['total_items']}")
        
        logger.info("\n" + "="*60)
        logger.info("📊 Column Data (sorted top to bottom):")
        logger.info("="*60 + "\n")
        
        for i, item in enumerate(result['column_data'], 1):
            bbox = item['bounding_box']
            logger.info(f"{i:2d}. {item['field_type']:<20}: {item['text']}")
            logger.info(f"    Position: ({bbox['left']:.0f}, {bbox['top']:.0f})")
            logger.info(f"    Size: {bbox['width']:.0f} x {bbox['height']:.0f}")
        
        # Column bounding box for highlighting
        col_bbox = result['column_bounding_box']
        logger.info("\n" + "="*60)
        logger.info("🎯 Column Bounding Box (for frontend highlighting):")
        logger.info("="*60)
        logger.info(f"   Left: {col_bbox['left']:.0f}")
        logger.info(f"   Top: {col_bbox['top']:.0f}")
        logger.info(f"   Width: {col_bbox['width']:.0f}")
        logger.info(f"   Height: {col_bbox['height']:.0f}")
        
        # Save results
        output_file = ocr_json_path.parent / f"{ocr_json_path.stem}_column_{serial_number}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Saved to: {output_file}")
        
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
    # ... (keep existing process_pdf function as is)
    pass


def list_input_files() -> None:
    """List all PDF files in the input directory"""
    # ... (keep existing list_input_files function as is)
    pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process PDF files with Azure Computer Vision OCR and LLM extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF with Azure OCR
  python main.py --file data/input/document.pdf
  
  # Process and search for terms
  python main.py --file data/input/document.pdf --search "862909" "B3219"
  
  # Use LLM to extract column data from existing OCR results
  python main.py --llm-only data/output/ocr_results/sample_page_1_ocr.json "862909"
  
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
        metavar=('OCR_JSON', 'SERIAL_NUMBER'),
        help='Use LLM to extract column data from existing OCR JSON file'
    )
    
    args = parser.parse_args()
    
    # Handle LLM-only mode
    if args.llm_only:
        ocr_json_path, serial_number = args.llm_only
        process_pdf_with_llm(ocr_json_path, serial_number)
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