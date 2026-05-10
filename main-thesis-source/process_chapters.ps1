$chap4_path = "C:\Users\admin\Desktop\pfe_preparation\main-thesis-source\chap4_logistics.tex"
$chap3_path = "C:\Users\admin\Desktop\pfe_preparation\main-thesis-source\chap3_safety.tex"

$chap4 = Get-Content $chap4_path -Raw
$chap3 = Get-Content $chap3_path -Raw

# Extract Forklift Detection Implementation
$regex = "(?s)\\section\{Forklift Detection Implementation\}(.*?)(?=\\section\{Speed and Angle Estimation\})"
if ($chap4 -match $regex) {
    $forklift_content = "\section{Forklift Detection Implementation}" + $matches[1]
    
    # Remove from chap4
    $chap4 = $chap4.Replace($forklift_content, "")
    
    # Extract inputs and remove them
    $yolov8 = "\input{chap_yolov8_architecture.tex}"
    $yolov5 = "\input{chap_yolov5_justification.tex}"
    
    $chap4 = $chap4.Replace($yolov8 + "`r`n", "")
    $chap4 = $chap4.Replace($yolov8 + "`n", "")
    $chap4 = $chap4.Replace($yolov8, "")
    
    $forklift_content = $forklift_content.Replace($yolov5 + "`r`n", "")
    $forklift_content = $forklift_content.Replace($yolov5 + "`n", "")
    $forklift_content = $forklift_content.Replace($yolov5, "")
    
    $sota_section = "`n\section{State of the Art: The YOLO Architectures}`nThis section details the architectures of the YOLO family models utilized in this phase, specifically justifying the selection of YOLOv5 and explaining the advancements in YOLOv8.`n`n\input{chap_yolov5_justification.tex}`n`n\input{chap_yolov8_architecture.tex}`n`n"
    
    $chap3 = $chap3.Replace("\section{Fire and Smoke Detection Implementation}", $sota_section + "\section{Fire and Smoke Detection Implementation}")
    $chap3 = $chap3.Replace("\section{Chapter Summary}", $forklift_content + "\section{Chapter Summary}")
    
    $chap3 = $chap3.Replace("Sprint Backlog For Sprint 2", "Implementation Objectives")
    $chap3 = $chap3.Replace("Design Diagrams For Sprint 2", "Design Diagrams")
    $chap3 = $chap3.Replace("Implementation Of Sprint 2", "Implementation Details")
    
    $chap4 = $chap4.Replace("Sprint Backlog For Sprint 1", "Implementation Objectives")
    $chap4 = $chap4.Replace("Design Diagrams For Sprint 1", "Design Diagrams")
    $chap4 = $chap4.Replace("Implementation Of Sprint 1", "Implementation Details")
    $chap4 = $chap4.Replace("during Sprint 1", "during this phase")
    $chap4 = $chap4.Replace("subsequent sprints", "subsequent phases")
    
    # Fix repeated \section{Speed and Angle Estimation} in chap4
    $chap4 = $chap4.Replace("\section{Speed and Angle Estimation}`r`n`r`n\section{Speed and Angle Estimation}", "\section{Speed and Angle Estimation}")
    $chap4 = $chap4.Replace("\section{Speed and Angle Estimation}`n`n\section{Speed and Angle Estimation}", "\section{Speed and Angle Estimation}")
    
    Set-Content $chap4_path $chap4
    Set-Content $chap3_path $chap3
    
    Write-Host "Success"
} else {
    Write-Host "Regex did not match"
}
