"""
PDF Parser with OCR support using pdfplumber and Tesseract
"""

import numpy as np  # Add this
import pdfplumber
import pytesseract
from PIL import Image
from typing import List, Dict, Optional
import io
import re


class PDFParser:
    """
    Parses PDF documents with support for:
    - Text extraction (pdfplumber)
    - OCR for scanned documents (Tesseract)
    - Layout detection (sections, headers, tables)
    """
    
    def __init__(self, use_ocr: bool = False, ocr_lang: str = 'eng'):
        """
        Args:
            use_ocr: Enable OCR for scanned PDFs
            ocr_lang: Tesseract language code
        """
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
    
    def parse(self, pdf_path: str) -> Dict:
        """
        Parse PDF document and extract text, layout, and metadata
        
        Returns:
            Dictionary with:
            - text: Full extracted text
            - pages: List of page dictionaries
            - layout: Detected structure (headers, sections, tables)
            - metadata: Document metadata
        """
        result = {
            'text': '',
            'pages': [],
            'layout': [],
            'metadata': {}
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                result['metadata'] = {
                    'num_pages': len(pdf.pages),
                    'title': pdf.metadata.get('Title', ''),
                    'author': pdf.metadata.get('Author', ''),
                    'subject': pdf.metadata.get('Subject', '')
                }
                
                # Extract text and layout from each page
                for page_num, page in enumerate(pdf.pages, 1):
                    page_data = self._extract_page_data(page, page_num)
                    result['pages'].append(page_data)
                    result['text'] += page_data['text'] + '\n\n'
                
                # Detect document structure
                result['layout'] = self._detect_structure(result['pages'])
        
        except Exception as e:
            # Fallback to OCR if pdfplumber fails
            if self.use_ocr:
                result = self._parse_with_ocr(pdf_path)
            else:
                raise Exception(f"Failed to parse PDF: {str(e)}")
        
        return result
    
    def _extract_page_data(self, page, page_num: int) -> Dict:
        """Extract text and layout from a single page"""
        page_data = {
            'page_num': page_num,
            'text': '',
            'tables': [],
            'images': [],
            'bbox': None
        }
        
        # Extract text
        text = page.extract_text()
        if not text and self.use_ocr:
            # Try OCR if text extraction failed
            text = self._ocr_page(page)
        
        page_data['text'] = text or ''
        
        # Extract tables
        tables = page.extract_tables()
        page_data['tables'] = tables if tables else []
        
        # Extract bounding box
        page_data['bbox'] = {
            'x0': page.bbox[0],
            'y0': page.bbox[1],
            'x1': page.bbox[2],
            'y1': page.bbox[3]
        }
        
        return page_data
    
    def _detect_structure(self, pages: List[Dict]) -> List[Dict]:
        """
        Detect document structure: sections, headers, etc.
        Simple heuristic-based approach (can be enhanced with ML)
        """
        structure = []
        
        for page in pages:
            text = page['text']
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Heuristic: Headers are short, often in caps, or numbered
                if self._is_likely_header(line, i, lines):
                    structure.append({
                        'type': 'header',
                        'text': line,
                        'page': page['page_num'],
                        'level': self._get_header_level(line)
                    })
        
        return structure
    
    def _is_likely_header(self, line: str, line_idx: int, all_lines: List[str]) -> bool:
        """Heuristic to detect headers"""
        # Short lines (likely headers)
        if len(line) < 100 and len(line.split()) < 15:
            # Check if next line is not empty (header followed by content)
            if line_idx + 1 < len(all_lines) and all_lines[line_idx + 1].strip():
                # Check for common header patterns
                if any(keyword in line.lower() for keyword in ['section', 'article', 'clause', 'chapter']):
                    return True
                # All caps (often headers)
                if line.isupper() and len(line) > 3:
                    return True
                # Numbered sections
                if any(line.strip().startswith(f'{i}.') for i in range(1, 20)):
                    return True
        return False
    
    def _get_header_level(self, line: str) -> int:
        """Determine header level (1-3)"""
        if line.strip().startswith(('CHAPTER', 'PART')):
            return 1
        elif line.strip().startswith(('SECTION', 'ARTICLE')):
            return 2
        else:
            return 3
    
    def _ocr_page(self, page) -> str:
        """Extract text using OCR (Tesseract)"""
        try:
            # Convert page to image
            im = page.to_image(resolution=300)
            # Perform OCR
            text = pytesseract.image_to_string(im.pil, lang=self.ocr_lang)
            return text
        except Exception as e:
            print(f"OCR failed: {str(e)}")
            return ''
    
    def _parse_with_ocr(self, pdf_path: str) -> Dict:
        """Fallback: Parse entire PDF with OCR"""
        # This is a simplified version - in production, use pdf2image
        result = {
            'text': '',
            'pages': [],
            'layout': [],
            'metadata': {}
        }
        # Implementation would use pdf2image + pytesseract
        return result

