"""
Main script for processing PDFs with Azure OCR and automatic LLM extraction
WORKFLOW: PDF → Azure OCR (bounding boxes) → LLM (structured extraction) → Database
"""
import argparse
import sys
from pathlib import Path
import logging
from typing import Optional

from src.pdf_processor import PDFProcessor
from src.azure_ocr import AzureOCR
from src.text_mapper import TextMapper
from src.database import DatabaseManager
from src.llm_field_extractor import LLMFieldExtractor
from config.config import DATABASE_URL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_pdf_full_pipeline(pdf_path: str, page_num: Optional[int] = None) -> None:
    """
    COMPLETE PIPELINE: PDF → Azure OCR → LLM Extraction → Database
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Specific page to process (None = all pages)
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        logger.error(f"❌ File not found: {pdf_path}")
        return
    
    logger.info("="*80)
    logger.info(f"🚀 FULL PIPELINE: Azure OCR → LLM Extraction → Database")
    logger.info(f"   File: {pdf_path.name}")
    if page_num:
        logger.info(f"   Page: {page_num}")
    logger.info("="*80)
    
    # Initialize database
    db = None
    try:
        db = DatabaseManager(DATABASE_URL)
        logger.info(f"   💾 Database connected: {DATABASE_URL}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        return
    
    try:
        # Initialize processors
        pdf_processor = PDFProcessor()
        azure_ocr = AzureOCR()
        text_mapper = TextMapper()
        llm_extractor = LLMFieldExtractor()
        
        # Convert PDF to images
        logger.info("\n🖼️  Step 1: Converting PDF to images...")
        images = pdf_processor.pdf_to_images(str(pdf_path))
        
        if page_num:
            if page_num > len(images):
                logger.error(f"❌ Page {page_num} not found (total pages: {len(images)})")
                return
            images = [images[page_num - 1]]
            logger.info(f"   Processing page {page_num}")
        else:
            logger.info(f"   Total pages: {len(images)}")
        
        # Save document to database
        document_id = db.save_document(
            filename=pdf_path.name,
            file_path=str(pdf_path.absolute()),
            total_pages=len(images)
        )
        logger.info(f"   📝 Document saved to DB (ID: {document_id})")
        
        # Process each page
        all_extraction_results = []
        
        for idx, image in enumerate(images, 1):
            current_page = page_num if page_num else idx
            logger.info(f"\n{'='*80}")
            logger.info(f"📄 Step 2: Processing Page {current_page}")
            logger.info(f"{'='*80}")
            
            # STEP 1: Perform Azure OCR to get bounding boxes
            logger.info("   🔍 Performing Azure OCR (getting bounding boxes)...")
            ocr_result = azure_ocr.analyze_image(image)
            
            if not ocr_result:
                logger.warning(f"   ⚠️  No OCR results for page {current_page}")
                continue
            
            # Map text blocks with bounding boxes
            text_blocks = text_mapper.map_text_blocks(ocr_result)
            
            # Build OCR data structure with all bounding boxes
            ocr_data = {
                'filename': pdf_path.name,
                'page_number': current_page,
                'text_blocks': text_blocks,
                'total_blocks': len(text_blocks),
                'image_dimensions': {
                    'width': image.width,
                    'height': image.height
                }
            }
            
            logger.info(f"   ✅ Azure OCR: Extracted {len(text_blocks)} text blocks with bounding boxes")
            logger.info(f"   📐 Image size: {image.width} x {image.height}")
            
            # Save OCR to database
            ocr_result_id = db.save_ocr_result(document_id, current_page, ocr_data)
            logger.info(f"   📝 OCR saved to DB (ID: {ocr_result_id})")
            
            # STEP 2: LLM extracts structured data using OCR bounding boxes
            logger.info(f"\n{'='*80}")
            logger.info(f"🧠 Step 3: LLM Structured Extraction (using OCR bounding boxes)")
            logger.info(f"{'='*80}")
            
            try:
                # LLM discovers identifiers and extracts data
                extraction_results = llm_extractor.extract_all_data(ocr_data)
                
                if not extraction_results:
                    logger.warning("   ⚠️  No identifiers found on this page")
                    continue
                
                # Save each extraction result to database
                for result in extraction_results:
                    identifier = result['identifier']
                    logger.info(f"\n   📊 Processing: {identifier}")
                    logger.info(f"      Type: {result['identifier_type']}")
                    logger.info(f"      Document Type: {result['document_type']}")
                    logger.info(f"      Fields extracted: {result['total_fields']}")
                    
                    # Save identifier to database
                    metadata = result.get('identifier_metadata', {})
                    identifier_id = db.save_identifier(
                        ocr_result_id=ocr_result_id,
                        identifier=identifier,
                        identifier_type=metadata.get('type', result['identifier_type']),
                        confidence=metadata.get('confidence', 1.0),
                        block_id=metadata.get('block_id', 0)
                    )
                    logger.info(f"      📝 Identifier saved to DB (ID: {identifier_id})")
                    
                    # Save extraction results with Pydantic models to database
                    extracted_data_id = db.save_extraction_result(identifier_id, result)
                    logger.info(f"      📝 Extraction saved to DB (ID: {extracted_data_id})")
                    
                    # Add to results list
                    all_extraction_results.append({
                        'page': current_page,
                        'identifier': identifier,
                        'result': result
                    })
            
            except Exception as e:
                logger.error(f"   ❌ Error during LLM extraction: {str(e)}", exc_info=True)
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*80}")
        logger.info(f"   📄 Pages processed: {len(images)}")
        logger.info(f"   🔍 Total identifiers found: {len(all_extraction_results)}")
        
        if all_extraction_results:
            logger.info(f"\n   📊 Extracted Identifiers:")
            for item in all_extraction_results:
                logger.info(f"      • {item['identifier']} (Page {item['page']})")
        
        logger.info(f"\n   💾 All data saved to database: {DATABASE_URL}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n❌ Error during pipeline: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        if db:
            db.close()


def query_database(identifier: Optional[str] = None) -> None:
    """Query database for extraction results"""
    try:
        db = DatabaseManager(DATABASE_URL)
        
        if identifier:
            logger.info(f"🔍 Searching for: {identifier}")
            results = db.search_by_identifier(identifier)
            
            if not results:
                logger.info(f"   ❌ No results found for '{identifier}'")
                return
            
            logger.info(f"   ✅ Found {len(results)} result(s)\n")
            
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. {result['filename']} (Page {result['page_number']})")
                logger.info(f"   Identifier: {result['identifier']} ({result['identifier_type']})")
                logger.info(f"   Document Type: {result['document_type']}")
                logger.info(f"   Fields: {result['extraction_json'].get('total_fields', 0)}")
                logger.info("")
        else:
            logger.info("📊 All Identifiers in Database:")
            identifiers = db.get_all_identifiers()
            
            if not identifiers:
                logger.info("   ❌ No identifiers found in database")
                return
            
            logger.info(f"   Total: {len(identifiers)}\n")
            
            for i, idf in enumerate(identifiers, 1):
                logger.info(f"{i:3d}. {idf['identifier']:20s} | {idf['identifier_type']:20s} | "
                          f"{idf['filename']:30s} | Page {idf['page_number']}")
        
        db.close()
        
    except Exception as e:
        logger.error(f"❌ Error querying database: {str(e)}")


def list_input_files() -> None:
    """List all PDF files in the input directory"""
    from config.config import INPUT_DIR
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
        description="Process PDF files with Azure OCR and LLM extraction (saves to database only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire PDF with auto-extraction
  python main.py --file data/sample.pdf
  
  # Process specific page
  python main.py --file data/sample.pdf --page 1
  
  # Query database
  python main.py --query "P-11217"
  python main.py --query-all
  
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
        '--list', '-l',
        action='store_true',
        help='List all PDF files in input directory'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Query database for specific identifier'
    )
    
    parser.add_argument(
        '--query-all',
        action='store_true',
        help='Show all identifiers in database'
    )
    
    args = parser.parse_args()
    
    # Handle database queries
    if args.query:
        query_database(identifier=args.query)
        return
    
    if args.query_all:
        query_database()
        return
    
    # Handle list command
    if args.list:
        list_input_files()
        return
    
    # Handle full pipeline
    if args.file:
        process_pdf_full_pipeline(
            pdf_path=args.file,
            page_num=args.page
        )
    else:
        # No arguments - show help
        parser.print_help()
        print("\n")
        list_input_files()


if __name__ == "__main__":
    main()