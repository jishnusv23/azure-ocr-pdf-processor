"""
Azure Computer Vision OCR Integration - Fixed
"""
import time
import json
from typing import List, Dict, Any
from pathlib import Path
import logging
from io import BytesIO

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image

from config.config import AZURE_CV_ENDPOINT, AZURE_CV_API_KEY, OCR_RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureOCR:
    """Azure Computer Vision OCR Service"""
    
    def __init__(self):
        """Initialize Azure Computer Vision client"""
        if not AZURE_CV_ENDPOINT or not AZURE_CV_API_KEY:
            raise ValueError("Azure credentials not configured")
        
        self.client = ComputerVisionClient(
            AZURE_CV_ENDPOINT,
            CognitiveServicesCredentials(AZURE_CV_API_KEY)
        )
        
        logger.info("✅ Azure Computer Vision client initialized")
    
    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze a PIL Image with Azure OCR (main method used by your code)
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing OCR results with text and positions
        """
        return self.process_image(image)
    
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process a PIL Image with Azure OCR
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing OCR results with text and positions
        """
        logger.info("🔍 Starting OCR processing...")
        
        # Convert PIL Image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        try:
            # Call Azure Read API (most accurate for documents)
            read_operation = self.client.read_in_stream(
                img_byte_arr,
                raw=True
            )
            
            # Get operation location
            operation_location = read_operation.headers["Operation-Location"]
            operation_id = operation_location.split("/")[-1]
            
            # Wait for the operation to complete
            logger.info("   ⏳ Waiting for OCR to complete...")
            max_retries = 30
            retry_count = 0
            
            while retry_count < max_retries:
                read_result = self.client.get_read_result(operation_id)
                
                if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                    break
                
                time.sleep(1)
                retry_count += 1
            
            if read_result.status == OperationStatusCodes.succeeded:
                logger.info("   ✅ OCR completed successfully")
                return self._parse_read_results(read_result, image.size)
            else:
                logger.error(f"   ❌ OCR failed with status: {read_result.status}")
                return {"error": f"OCR failed: {read_result.status}"}
                
        except Exception as e:
            logger.error(f"❌ Error during OCR: {str(e)}")
            raise
    
    def _parse_read_results(self, read_result, image_size: tuple) -> Dict[str, Any]:
        """
        Parse Azure Read API results into structured format
        
        Args:
            read_result: Azure API response
            image_size: (width, height) of the image
            
        Returns:
            Structured dictionary with text and positions
        """
        width, height = image_size
        
        results = {
            "image_dimensions": {
                "width": width, 
                "height": height
            },
            "pages": [],
            "all_text": "",
            "text_blocks": []
        }
        
        for page_idx, page in enumerate(read_result.analyze_result.read_results):
            page_data = {
                "page_number": page_idx + 1,
                "width": page.width,
                "height": page.height,
                "lines": []
            }
            
            for line in page.lines:
                # Get bounding box (8 coordinates: x1,y1, x2,y2, x3,y3, x4,y4)
                bbox = line.bounding_box
                
                # Convert to simple rectangle (left, top, right, bottom)
                left = min(bbox[0], bbox[6])
                top = min(bbox[1], bbox[3])
                right = max(bbox[2], bbox[4])
                bottom = max(bbox[5], bbox[7])
                
                line_data = {
                    "text": line.text,
                    "bounding_box": {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "width": right - left,
                        "height": bottom - top
                    },
                    "words": []
                }
                
                # Process words in the line
                for word in line.words:
                    word_bbox = word.bounding_box
                    word_left = min(word_bbox[0], word_bbox[6])
                    word_top = min(word_bbox[1], word_bbox[3])
                    word_right = max(word_bbox[2], word_bbox[4])
                    word_bottom = max(word_bbox[5], word_bbox[7])
                    
                    word_data = {
                        "text": word.text,
                        "confidence": word.confidence,
                        "bounding_box": {
                            "left": word_left,
                            "top": word_top,
                            "right": word_right,
                            "bottom": word_bottom,
                            "width": word_right - word_left,
                            "height": word_bottom - word_top
                        }
                    }
                    
                    line_data["words"].append(word_data)
                    results["text_blocks"].append(word_data)
                
                page_data["lines"].append(line_data)
                results["all_text"] += line.text + "\n"
            
            results["pages"].append(page_data)
        
        logger.info(f"   📊 Extracted {len(results['text_blocks'])} text blocks")
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save OCR results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved OCR results: {output_path}")
    
    def load_results(self, json_path: str) -> Dict[str, Any]:
        """Load OCR results from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


if __name__ == "__main__":
    # Test Azure OCR
    from src.pdf_processor import PDFProcessor
    from config.config import INPUT_DIR
    
    # Find first PDF in input directory
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    if pdf_files:
        test_pdf = pdf_files[0]
        logger.info(f"\n🧪 Testing with: {test_pdf.name}")
        
        # Convert PDF to image
        processor = PDFProcessor()
        images = processor.pdf_to_images(str(test_pdf))
        
        if images:
            # Process first page with OCR
            image = images[0]
            
            ocr = AzureOCR()
            results = ocr.analyze_image(image)
            
            # Save results
            output_path = OCR_RESULTS_DIR / f"{test_pdf.stem}_page_1_ocr.json"
            ocr.save_results(results, str(output_path))
            
            # Print summary
            print(f"\n📊 OCR Results Summary:")
            print(f"   Total text blocks: {len(results['text_blocks'])}")
            print(f"   First 3 text blocks:")
            for block in results['text_blocks'][:3]:
                print(f"      - '{block['text']}' (confidence: {block.get('confidence', 'N/A')})")
    else:
        logger.warning(f"⚠️  No PDF files found in: {INPUT_DIR}")