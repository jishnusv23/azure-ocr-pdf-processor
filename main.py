"""
Main script for processing PDFs with Azure OCR
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
from config.config import INPUT_DIR, OCR_RESULTS_DIR, VISUALIZATIONS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        logger.error(f"❌ PDF file not found: {pdf_path}")
        return
    
    logger.info("="*60)
    logger.info(f"🚀 Starting PDF OCR Processing")
    logger.info(f"   File: {pdf_path.name}")
    logger.info("="*60)
    
    try:
        # Step 1: Convert PDF to images
        logger.info("\n📄 Step 1: Converting PDF to images...")
        processor = PDFProcessor()
        images = processor.convert_pdf_to_images(str(pdf_path))
        
        # Get PDF info
        pdf_info = processor.get_pdf_info(str(pdf_path))
        
        # Filter to specific page if requested
        if page_num is not None:
            images = [(num, img) for num, img in images if num == page_num]
            if not images:
                logger.error(f"❌ Page {page_num} not found in PDF")
                return
        
        # Step 2: Initialize Azure OCR
        logger.info("\n🔍 Step 2: Initializing Azure OCR...")
        ocr = AzureOCR()
        
        # Step 3: Initialize database (if needed)
        db = None
        doc_id = None
        if save_to_db:
            logger.info("\n💾 Step 3: Initializing database...")
            db = DatabaseManager()
            if db.engine:
                db.create_tables()
                doc_id = db.save_document(
                    filename=pdf_path.name,
                    file_path=str(pdf_path),
                    total_pages=pdf_info['total_pages'],
                    file_size_mb=pdf_info['file_size_mb'],
                    metadata=pdf_info['metadata']
                )
        
        # Step 4: Process each page
        logger.info("\n⚙️  Step 4: Processing pages with OCR...")
        
        all_search_results = {}
        
        for page_num, image in images:
            logger.info(f"\n   Processing page {page_num}...")
            
            # Run OCR
            results = ocr.process_image(image)
            
            # Save results to JSON
            output_filename = f"{pdf_path.stem}_page_{page_num}_ocr.json"
            output_path = OCR_RESULTS_DIR / output_filename
            ocr.save_results(results, str(output_path))
            
            # Save visualization
            vis_filename = f"{pdf_path.stem}_page_{page_num}.png"
            vis_path = VISUALIZATIONS_DIR / vis_filename
            processor.save_image(image, str(vis_path))
            
            # Save to database
            if db and doc_id:
                db.save_ocr_result(doc_id, page_num, results)
            
            # Search for terms if provided
            if search_terms:
                mapper = TextMapper(results)
                page_search_results = mapper.search_multiple(search_terms)
                
                for term, matches in page_search_results.items():
                    if term not in all_search_results:
                        all_search_results[term] = []
                    
                    for match in matches:
                        match['page'] = page_num
                        all_search_results[term].append(match)
            
            # Print summary
            logger.info(f"   ✅ Page {page_num} completed:")
            logger.info(f"      - Text blocks extracted: {len(results['text_blocks'])}")
            logger.info(f"      - OCR results: {output_path}")
            logger.info(f"      - Image saved: {vis_path}")
        
        # Print search results
        if search_terms and all_search_results:
            logger.info("\n" + "="*60)
            logger.info("🔍 Search Results:")
            for term, matches in all_search_results.items():
                logger.info(f"\n   '{term}': {len(matches)} matches")
                for match in matches[:3]:  # Show first 3
                    bbox = match['bounding_box']
                    logger.info(f"      - Page {match['page']}: Position ({bbox['left']:.0f}, {bbox['top']:.0f})")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 Processing completed successfully!")
        logger.info(f"   Total pages processed: {len(images)}")
        logger.info(f"   Results directory: {OCR_RESULTS_DIR}")
        if save_to_db and doc_id:
            logger.info(f"   Database document ID: {doc_id}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n❌ Error during processing: {str(e)}", exc_info=True)
        sys.exit(1)


def list_input_files() -> None:
    """List all PDF files in the input directory"""
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    if not pdf_files:
        logger.info(f"📂 No PDF files found in: {INPUT_DIR}")
        logger.info(f"   Please place PDF files in the input directory")
        return
    
    logger.info(f"\n📂 PDF files in input directory:")
    for i, pdf in enumerate(pdf_files, 1):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        logger.info(f"   {i}. {pdf.name} ({size_mb:.2f} MB)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process PDF files with Azure Computer Vision OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a specific PDF file
  python main.py --file data/input/document.pdf
  
  # Process only page 2 of a PDF
  python main.py --file data/input/document.pdf --page 2
  
  # Process and save to database
  python main.py --file data/input/document.pdf --db
  
  # Process and search for terms
  python main.py --file data/input/document.pdf --search "862909" "B3219"
  
  # List all PDF files in input directory
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
    
    args = parser.parse_args()
    
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