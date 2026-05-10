const fs = require('fs');

const patchStr = `chap5_zeroshot.tex:1:\\chapter{Release 4: Zero-Shot Shelf Monitoring}
chap5_zeroshot.tex:5:This chapter presents the technical background and implementation work for the retail inventory capabilities within the EYE-D video analytics solution. The work described here corresponds to Release 4 of the project, which focused on a Zero-Shot Shelf Fullness Detection system. In contrast to previous supervised learning approaches (e.g., forklift detection), this release leverages advanced Vision-Language Models (VLMs) to detect objects based on natural language prompts without requiring a single annotated bounding box for that specific class. The primary goal is to make zero-shot models work on specific hardware constraints (NPU) for broader use cases, heavily emphasizing ONNX exportation and optimized inference engines customized for users' specific operational conditions.
chap5_zeroshot.tex:63:The conceptual architecture of Release 3 is anchored in the theory of \\textbf{Negative Space}. In highly variable retail environments, tracking the presence of every unique product is computationally and logistically prohibitive. Negative Space theory inverts this problem: the system focuses on detecting the structural substrate of the shelf and the explicit absence of products (voids). By quantifying the empty spaces against the known physical boundaries of the shelving unit, the system deduces inventory levels deterministically, rendering it immune to SKU variation.
chap5_zeroshot.tex:138:By fusing high-accuracy VLMs with robust edge-optimization strategies, the system delivers an air-gapped, fault-tolerant solution capable of real-time inventory monitoring. The resilient fallback mechanism ensures that the theoretical benefits of semantic grounding are practically viable within the strict constraints of industrial NPUs. The most important point of Release 4 was to verify that the zero-shot model could run effectively on the hardware, proving that clients of the device could easily customize their conditions and dynamically monitor different products using natural language, all without the need for exhaustive data collection or re-training.
chap5_zeroshot.tex:140:Furthermore, this deployment represents the culmination of the project's MLOps lifecycle. To achieve optimal performance on the target NPU, ONNX INT8 quantization was systematically applied to the weights. This process completes the "Full Circle" of the project's engineering methodology: starting from the rigorous data collection and engineering of the 22,607-image dataset (\\textbf{Release 1}), progressing through the supervised fine-tuning and hard-negative mining for safety and logistics (\\textbf{Releases 2 \\& 3}), and culminating in the NPU optimization and deployment of advanced zero-shot capabilities (\\textbf{Release 4}). This comprehensive approach ensures that all models—whether fine-tuned or open-vocabulary—are deeply aligned with the specific hardware constraints and operational realities of the host organization.
chap4_logistics.tex:1:\\chapter{Release 3: Logistical Optimization (Forklift Tracking \\& Analytics)}
chap4_logistics.tex:5:This chapter presents the technical background and implementation work for logistical optimization within the EYE-D video analytics solution. The work described here corresponds to Release 3 of the project, focusing on training a custom object detection model for forklifts, and developing foundational components for measuring object speed and heading direction from video streams.
chap4_logistics.tex:8:The logistical optimization models require highly specific adaptations to function effectively in real-world scenarios. \\textbf{While base weights from YOLOv8n/v5n were used, the final weights were fine-tuned using the 22,607-image industrial dataset described in Release 1 to ensure high accuracy on the specific lighting and camera angles of the host organization.} Crucially, the Forklift images from Batch 3 (newcam) were specifically isolated and used to fine-tune the tracking model. This targeted refinement directly connects the rigorous data engineering of Release 1 to the logistical tracking accuracy achieved in this phase.
chap3_safety.tex:1:\\chapter{Release 2: Industrial Safety (Fire, Smoke \\& PPE Detection)}
chap3_safety.tex:5:This chapter presents the technical background and implementation work for the industrial safety capabilities within the EYE-D video analytics solution. Designated as Release 2 of the project, this phase focuses on developing and deploying a robust fire and smoke detection system to trigger immediate emergency alerts, prioritizing real-time responsiveness and the elimination of false positive alarms. Environmental hazards like fire pose immediate existential threats to both personnel and infrastructure, demanding a highly optimized inference model capable of running natively on edge Neural Processing Units (NPUs).
chap3_safety.tex:10:The core component of Release 2 focuses on early hazard detection, specifically identifying fire and smoke in industrial environments. This implementation prioritizes real-time alerting while drastically reducing false positives (e.g., misclassifying steam, reflections, or intense industrial lighting as fire).
chap3_safety.tex:15:\\textbf{While base weights from YOLOv8n were used, the final weights were fine-tuned using the 22,607-image industrial dataset described in Release 1 to ensure high accuracy on the specific lighting and camera angles of the host organization.}
chap3_safety.tex:27:To resolve this, the system required an aggressive \\textbf{Hard-Negative Mining} strategy. Initial models flagged industrial lights as fire. Through the Hard-Negative Mining process documented in Release 1, these images were re-labeled as background (fed back into the training loop with empty \\texttt{.txt} label files). This iterative refinement allowed the model to 'learn' the difference between glare and combustion. Furthermore, we extracted approximately 800 ''hard negative'' images from the COCO validation dataset \\cite{ref:coco}, specifically targeting classes known to cause confusion (traffic lights, lamps, TVs).
chap3_safety.tex:54:This chapter detailed the execution of Release 2, transitioning the EYE-D system toward comprehensive industrial safety capabilities. The fire and smoke detection module successfully incorporated hard-negative mining on the YOLOv8n architecture to aggressively minimize false positive alerts in complex warehouse environments.
chap3_safety.tex:56:The implementation maintained strict adherence to NPU hardware constraints. By standardizing the export pipeline to ONNX and compiling to optimized inference engines, Release 2 successfully bridged the gap between rigorous model training in PyTorch and real-time edge deployment. This directly fulfills the project's mandate for responsive, localized industrial hazard monitoring.
chap2_data.tex:1:\\chapter{Release 1: Dataset Lifecycle \\& Engineering}
chap2_data.tex:5:The foundation of any robust Machine Learning Operations (MLOps) pipeline is data. This chapter outlines the Dataset Lifecycle \\& Engineering phase, designated as Release 1 of the project. Before any advanced computer vision models could be trained or deployed on industrial Neural Processing Units (NPUs), a massive, domain-specific dataset had to be engineered. This chapter details the collection, pre-annotation, manual validation, and final exportation of a custom industrial dataset comprising over 22,600 images.
chap2_data.tex:65:The engineering of this 22,607-image dataset proves that the project follows a rigorous MLOps lifecycle. By establishing a robust data foundation in Release 1, all subsequent fine-tuning, safety monitoring (Release 2), logistical tracking (Release 3), and NPU optimizations are anchored in high-quality, domain-specific visual data.
chap_yolov8_architecture.tex:165:The quantitative comparison used in Release 1 is reported in Table~\\ref{tab:yolov8-variants}.
chap_yolov8_architecture.tex:343:However, the architectural sophistication of YOLOv8---particularly the Task-Aligned Assigner, decoupled head, and C2f module---introduces additional computational overhead that may not be justified for all deployment scenarios. The next release (Sprint 2) presents a detailed justification for why YOLOv5 was selected over YOLOv8 for the single-class detection tasks in this project.
chap_yolov5_justification.tex:5:This section provides a comprehensive technical justification for selecting YOLOv5 over YOLOv8 as the detection framework for the custom-trained models deployed in EYE-D. Although YOLOv8 offers architectural improvements (as detailed in the Release 1 state of the art), it introduces additional computational complexity that is unnecessary---and potentially counterproductive---for the single-class detection scenarios targeted by this project.
chap1.tex:145:The project transitioned from a Scrum-based management description to the CRISP-DM (Cross-Industry Standard Process for Data Mining) technical framework to formally capture the engineering maturity of the solution. Both Release 1 (Forklift Detection and Speed Estimation) and Release 2 (Fire and Smoke Detection) cycled through the following six phases:
chap1.tex:148:    \\item \\textbf{Phase 1: Business Understanding:} We defined the core operational needs for the EYE-D system. For Release 1, the objective was optimizing warehouse logistics and traffic management via Forklift tracking and speed estimation. For Release 2, the focus shifted to industrial safety, requiring real-time fire and smoke detection to trigger immediate emergency alerts.
chap1.tex:149:    \\item \\textbf{Phase 2: Data Understanding:} High-quality data acquisition was critical. Release 1 utilized a custom, curated Forklift dataset to address the absence of industrial vehicles in standard COCO datasets. Release 2 leveraged the HuggingFace \\texttt{fire-smoke-hardnegatives-int8} dataset, emphasizing the need to understand environmental hazards and challenging lighting conditions.
chap1.tex:150:    \\item \\textbf{Phase 3: Data Preparation:} Advanced data augmentations were applied to ensure generalization in real-world industrial environments. For both releases, techniques such as Mosaic and Mixup were integrated. Crucially, Release 2 incorporated hard-negative mining to actively reduce false positives (e.g., misclassifying steam or reflections as fire).
chap1.tex:151:    \\item \\textbf{Phase 4: Modeling:} Architecture selection was strictly guided by NPU hardware constraints. For Release 1, YOLOv5 was chosen for its proven low-latency NPU optimization. For Release 2, YOLOv8n was selected to leverage its advanced multi-scale feature extraction while remaining compact enough for edge deployment.`;

const lines = patchStr.trim().split('\n');
const fileData = {};

for (const line of lines) {
  const parts = line.split(':');
  if (parts.length < 3) continue;
  const fileName = parts[0];
  const lineNumber = parseInt(parts[1], 10);
  const originalLine = parts.slice(2).join(':');

  if (!fileData[fileName]) {
    fileData[fileName] = fs.readFileSync(fileName, 'utf8').split('\n');
  }

  let targetContent = originalLine;
  // Apply our intended transformation
  targetContent = targetContent.replace(/\bRelease (\d)\b/g, 'Phase $1');
  targetContent = targetContent.replace(/\bReleases (\d)\b/g, 'Phases $1');
  targetContent = targetContent.replace(/\brelease (\d)\b/g, 'phase $1');
  targetContent = targetContent.replace(/\breleases (\d)\b/g, 'phases $1');
  targetContent = targetContent.replace(/\bRelease\b(?!\s\d)/g, 'Phase');
  targetContent = targetContent.replace(/For both releases/g, 'For both phases');
  targetContent = targetContent.replace(/next release \(Sprint 2\)/g, 'next chapter');
  
  // Actually, wait, the user asked to replace "release" with "phase" or "chapter".
  targetContent = targetContent.replace(/\brelease\b(?!s)/gi, 'phase');
  // Wait, I already did the specific ones. Just re-running my regexes.
  
  if (lineNumber - 1 < fileData[fileName].length) {
    fileData[fileName][lineNumber - 1] = targetContent;
  }
}

for (const [fileName, fileLines] of Object.entries(fileData)) {
  fs.writeFileSync(fileName, fileLines.join('\n'), 'utf8');
}
console.log('Fixed files successfully!');
