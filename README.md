# Evaluating YOLOv8 Effectiveness in Detecting and Counting Overlapping Heads in Dense Crowd Images
## Strategies for Enhancing Performance in Occlusion-Heavy Scenarios
**Aims:**
is to assess the effectiveness of YOLOv8 in detecting and counting overlapping heads in dense crowd images. Additionally, to develop and evaluate strategies to improve performance in Occlusion-Heavy Scenarios.\
**Objectives:**\
To achieve the goals of this project, several objectives were considered to address the research
question proposed in section 1.2.1 above, as outlined below in this section:
- To quantify YOLOv8’s detection and counting accuracy (e.g., mAP, precision, recall,
and counting error) for overlapping heads in dense crowd images under varying levels of
occlusion.
- To identify limitations of YOLOv8 in occlusion-heavy scenarios using datasets extracted
from the Bradford University Event.
- To investigate and implement techniques such as Soft-NMS, multi-scale feature enhance-
ment, and higher resolution inputs to improve the detection of overlapping heads.
- To fine-tune YOLOv8 on a custom dataset with dense, occluded head annotations to
improve its generalisation to real-world crowd scenarios.
- To adjust hyperparameters (e.g., IoU threshold, confidence threshold) and model variants
(e.g., YOLOv8m vs. YOLOv8x) to optimize performance for small, overlapping objects.
- To compare the baseline performance of YOLOv8 with its enhanced versions (post-
strategy implementation) in terms of detection rates, false positives/negatives, and com-
putational efficiency.
- To provide actionable recommendations for deploying YOLOv8 in crowd monitoring sys-
tems, such as surveillance or event management.
