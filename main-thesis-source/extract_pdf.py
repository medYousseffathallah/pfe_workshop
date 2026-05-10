import sys
import os

pdf_path = r"c:\Users\admin\Desktop\pfe_preparation\old_repports\maincorrectphase1.pdf"
out_path = r"c:\Users\admin\Desktop\pfe_preparation\main-thesis-source\extracted_phase1.txt"

def extract_pdf():
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Extracted with PyPDF2")
        return True
    except ImportError:
        pass
    except Exception as e:
        print("PyPDF2 error:", e)

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Extracted with PyMuPDF")
        return True
    except ImportError:
        pass
    except Exception as e:
        print("PyMuPDF error:", e)
        
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Extracted with pdfplumber")
        return True
    except ImportError:
        pass
    except Exception as e:
        print("pdfplumber error:", e)

    print("Failed to extract: No suitable PDF library found.")
    return False

if __name__ == "__main__":
    extract_pdf()
