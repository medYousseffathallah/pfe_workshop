import re

def process_files():
    # Read chap4
    with open('C:/Users/admin/Desktop/pfe_preparation/main-thesis-source/chap4_logistics.tex', 'r', encoding='utf-8') as f:
        chap4_content = f.read()
        
    # Read chap3
    with open('C:/Users/admin/Desktop/pfe_preparation/main-thesis-source/chap3_safety.tex', 'r', encoding='utf-8') as f:
        chap3_content = f.read()

    # Extract Forklift Detection Implementation from chap4
    # It starts at \section{Forklift Detection Implementation}
    # And ends before \section{Speed and Angle Estimation}
    
    forklift_match = re.search(r'\\section\{Forklift Detection Implementation\}(.*?)(?=\\section\{Speed and Angle Estimation\})', chap4_content, re.DOTALL)
    if not forklift_match:
        print("Could not find Forklift section in chap4")
        return
        
    forklift_content = "\\section{Forklift Detection Implementation}" + forklift_match.group(1)
    
    # Remove forklift_content from chap4
    chap4_content = chap4_content.replace(forklift_content, "")
    
    # Extract \input{chap_yolov8_architecture.tex} from chap4 and remove it
    yolov8_input = "\\input{chap_yolov8_architecture.tex}\n"
    chap4_content = chap4_content.replace(yolov8_input, "")
    chap4_content = chap4_content.replace("\\input{chap_yolov8_architecture.tex}", "")
    
    # Also extract \input{chap_yolov5_justification.tex} from forklift_content and remove it
    # We will group them into a single State of the Art section in chap3
    yolov5_input = "\\input{chap_yolov5_justification.tex}\n"
    forklift_content = forklift_content.replace(yolov5_input, "")
    forklift_content = forklift_content.replace("\\input{chap_yolov5_justification.tex}", "")
    
    # Create the State of the Art section for chap3
    sota_section = """
\\section{State of the Art: The YOLO Architectures}
This section details the architectures of the YOLO family models utilized in this phase, specifically justifying the selection of YOLOv5 and explaining the advancements in YOLOv8.

\\input{chap_yolov5_justification.tex}

\\input{chap_yolov8_architecture.tex}

"""
    
    # Insert into chap3 before Fire and Smoke Detection Implementation
    # \section{Fire and Smoke Detection Implementation}
    chap3_content = chap3_content.replace("\\section{Fire and Smoke Detection Implementation}", sota_section + "\\section{Fire and Smoke Detection Implementation}")
    
    # Append Forklift Detection Implementation before Chapter Summary
    chap3_content = chap3_content.replace("\\section{Chapter Summary}", forklift_content + "\\section{Chapter Summary}")
    
    # Clean up chap4 (Sprint terminology)
    # The user asked to remove "Sprint" terminology earlier, but maybe it wasn't fully done? Let's replace Sprint with Phase/Chapter just in case
    # "Sprint Backlog For Sprint 2" -> "Objectives for Forklift Detection"
    # "Sprint Backlog For Sprint 1" -> "Objectives for Speed Estimation"
    chap3_content = chap3_content.replace("Sprint Backlog For Sprint 2", "Implementation Objectives")
    chap3_content = chap3_content.replace("Design Diagrams For Sprint 2", "Design Diagrams")
    chap3_content = chap3_content.replace("Implementation Of Sprint 2", "Implementation Details")
    
    chap4_content = chap4_content.replace("Sprint Backlog For Sprint 1", "Implementation Objectives")
    chap4_content = chap4_content.replace("Design Diagrams For Sprint 1", "Design Diagrams")
    chap4_content = chap4_content.replace("Implementation Of Sprint 1", "Implementation Details")
    chap4_content = chap4_content.replace("during Sprint 1", "during this phase")
    chap4_content = chap4_content.replace("subsequent sprints", "subsequent phases")
    
    # Write back
    with open('C:/Users/admin/Desktop/pfe_preparation/main-thesis-source/chap4_logistics.tex', 'w', encoding='utf-8') as f:
        f.write(chap4_content)
        
    with open('C:/Users/admin/Desktop/pfe_preparation/main-thesis-source/chap3_safety.tex', 'w', encoding='utf-8') as f:
        f.write(chap3_content)

    print("Successfully processed chap3 and chap4")

process_files()
