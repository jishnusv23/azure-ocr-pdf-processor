"""
Main script for processing PDFs with Azure OCR and automatic LLM extraction
WORKFLOW: PDF → Azure OCR → LLM Extraction → Database
NEW: Query and highlight specific identifier data in PDF
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
from config.config import DATABASE_URL, OUTPUT_DIR

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
    standalone_extraction.py
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


def highlight_identifier(identifier: str, output_dir: str = None) -> None:
    """
    Query database for identifier and create highlighted PDF
    
    Args:
        identifier: The identifier to search for (e.g., "P-11217", "A-7575")
        output_dir: Directory to save highlighted PDF (default: OUTPUT_DIR from config)
    """
    try:
        db = DatabaseManager(DATABASE_URL)
        
        logger.info("="*80)
        logger.info(f"🔍 HIGHLIGHT MODE: Searching for '{identifier}'")
        logger.info("="*80)
        
        # Search for identifier
        results = db.search_by_identifier(identifier)
        
        if not results:
            logger.error(f"   ❌ No results found for '{identifier}'")
            logger.info("\n💡 TIP: Use --query-all to see all available identifiers")
            return
        
        logger.info(f"   ✅ Found {len(results)} result(s)")
        
        # If duplicates, use the most recent one
        if len(results) > 1:
            logger.warning(f"   ⚠️  Found {len(results)} duplicate entries, using the most recent one")
            result = results[-1]  # Use the last (most recent) one
        else:
            result = results[0]
        
        filename = result['filename']
        page_number = result['page_number']
        extraction_json = result['extraction_json']
        
        logger.info(f"\n📄 Processing:")
        logger.info(f"   File: {filename}")
        logger.info(f"   Page: {page_number}")
        logger.info(f"   Identifier: {result['identifier']} ({result['identifier_type']})")
        logger.info(f"   Document Type: {result['document_type']}")
        logger.info("")
        
        # Get the original PDF path from database
        pdf_info = db.get_document_by_filename(filename)
        if not pdf_info:
            logger.error(f"   ❌ Could not find PDF info for {filename}")
            return
        
        pdf_path = Path(pdf_info['file_path'])
        if not pdf_path.exists():
            logger.error(f"   ❌ PDF file not found: {pdf_path}")
            return
        
        # Get OCR results for proper coordinate scaling
        ocr_data = db.get_ocr_by_document_and_page(pdf_info['id'], page_number)
        if not ocr_data:
            logger.error(f"   ❌ No OCR data found for page {page_number}")
            return
        
        ocr_results = json.loads(ocr_data['ocr_json'])
        
        # Prepare image_size for highlighter (it expects this format)
        ocr_results['image_size'] = {
            'width': ocr_data['image_width'],
            'height': ocr_data['image_height']
        }
        
        # CRITICAL FIX: Restructure extraction_json to have proper nested structure
        # The fields need to be in the format the highlighter expects
        fields = extraction_json.get('fields', {})
        logger.info(f"🔍 Debug: Found {len(fields)} top-level fields")
        logger.info(f"   Fields: {list(fields.keys())[:5]}...")  # Show first 5 field names
        
        # Set output directory
        if not output_dir:
            output_dir = OUTPUT_DIR
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_filename = f"{pdf_path.stem}_{identifier}_highlighted.pdf"
        output_path = output_dir / output_filename
        
        logger.info("\n🎨 Generating highlighted PDF...")
        logger.info(f"   Method: Yellow rectangles with borders")
        logger.info(f"   Output: {output_path}")
        logger.info("")
        
        # Highlight the PDF
        highlighted_pdf = highlight_extraction_in_pdf(
            pdf_path=str(pdf_path),
            extraction_result=extraction_json,
            output_path=str(output_path),
            method="rectangle",
            ocr_results=ocr_results
        )
        
        logger.info("="*80)
        logger.info("✅ SUCCESS!")
        logger.info("="*80)
        logger.info(f"📄 Highlighted PDF created: {highlighted_pdf}")
        logger.info(f"🎯 Showing data for: {result['identifier']}")
        logger.info(f"📊 Fields highlighted: {extraction_json.get('total_fields', 0)}")
        logger.info("="*80)
        
        db.close()
        
    except Exception as e:
        logger.error(f"❌ Error highlighting identifier: {str(e)}", exc_info=True)


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


def clean_database() -> None:
    """Clean all data from database tables"""
    try:
        db = DatabaseManager(DATABASE_URL)
        
        logger.info("="*80)
        logger.info("🗑️  DATABASE CLEANUP")
        logger.info("="*80)
        logger.info("⚠️  WARNING: This will delete ALL data from the database!")
        
        confirm = input("Are you sure? Type 'yes' to confirm: ")
        
        if confirm.lower() != 'yes':
            logger.info("❌ Cleanup cancelled")
            return
        
        with db.get_session() as session:
            from src.database import (Document, OCRResult, Identifier, 
                                     ExtractedData, ComponentTable, 
                                     StandaloneAssetTable, FlightInfoTable)
            
            # Delete in reverse order of dependencies
            deleted_components = session.query(ComponentTable).delete()
            deleted_standalone = session.query(StandaloneAssetTable).delete()
            deleted_flight = session.query(FlightInfoTable).delete()
            deleted_extracted = session.query(ExtractedData).delete()
            deleted_identifiers = session.query(Identifier).delete()
            deleted_ocr = session.query(OCRResult).delete()
            deleted_documents = session.query(Document).delete()
            
            session.commit()
            
            logger.info(f"✅ Deleted {deleted_components} component records")
            logger.info(f"✅ Deleted {deleted_standalone} standalone asset records")
            logger.info(f"✅ Deleted {deleted_flight} flight info records")
            logger.info(f"✅ Deleted {deleted_extracted} extraction records")
            logger.info(f"✅ Deleted {deleted_identifiers} identifiers")
            logger.info(f"✅ Deleted {deleted_ocr} OCR results")
            logger.info(f"✅ Deleted {deleted_documents} documents")
            logger.info("="*80)
            logger.info("✅ Database cleaned successfully!")
            logger.info("="*80)
        
        db.close()
        
    except Exception as e:
        logger.error(f"❌ Error cleaning database: {str(e)}")


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
  
  # Highlight specific identifier in PDF
  python main.py --highlight "P-11217"
  python main.py --highlight "A-7575"
  python main.py --highlight "862909"
  
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
    
    parser.add_argument(
        '--highlight', '-H',
        type=str,
        help='Highlight specific identifier in PDF (creates new PDF with highlights)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for highlighted PDFs (default: data/output)'
    )
    
    args = parser.parse_args()
    
    # Handle highlight mode (NEW FEATURE)
    if args.highlight:
        highlight_identifier(args.highlight, args.output_dir)
        return
    
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