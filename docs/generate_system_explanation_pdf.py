#!/usr/bin/env python3
"""Generate Insolare Safety System explanation PDF."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "Insolare_Safety_System_Explanation.pdf")


class SystemPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Insolare Safety System - Technical Overview", align="C")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.ln(4)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 120)
        self.multi_cell(self.epw, 8, title)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def subsection_title(self, title):
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.multi_cell(self.epw, 7, title)
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(self.epw, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(self.epw, 5.5, f"  - {text}")
        self.ln(1)

    def table_row(self, cols, widths, bold=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        for col, w in zip(cols, widths):
            self.cell(w, 7, col, border=1)
        self.ln()


def build_pdf():
    pdf = SystemPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)

    # Cover page
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 15, "Insolare Safety System", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, "Technical Overview", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 7, (
        "This document explains how the Insolare Safety System works: "
        "employee face recognition, PPE kit detection, live camera monitoring, "
        "image analysis, and backend data storage."
    ), align="C")
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Generated from project documentation", align="C")

    pdf.add_page()
    pdf.section_title("1. Overview")
    pdf.body_text(
        "The Insolare Safety System is a three-tier safety monitoring application that answers "
        "two questions at once:"
    )
    pdf.bullet("Who is this person? (face recognition)")
    pdf.bullet("Are they wearing the required PPE kit? (helmet, gloves, boots, jacket/vest)")
    pdf.ln(2)
    pdf.body_text(
        "The system combines a React frontend, a Node.js backend, and a Python Flask ML server "
        "to monitor construction-site safety in real time."
    )

    pdf.section_title("2. System Architecture")
    pdf.subsection_title("Three Main Components")
    w = [45, 125]
    pdf.table_row(["Layer", "Role"], w, bold=True)
    pdf.table_row(["Frontend (React, port 5173)", "UI for live camera, image upload, employee admin, dashboard"], w)
    pdf.table_row(["Backend (Node.js, port 3000)", "Auth, employee CRUD, stores detection events and PPE stats in Supabase"], w)
    pdf.table_row(["Flask Server (port 5000)", "All computer vision: face recognition, PPE detection, anti-spoofing"], w)
    pdf.ln(4)

    pdf.subsection_title("Data Flow")
    pdf.bullet("Frontend Camera Monitor connects to Flask MJPEG stream at /video_feed")
    pdf.bullet("Image Analysis uploads photos to Flask /api/analyze-image")
    pdf.bullet("Employee photos are saved by the backend and embeddings are generated via createEmbeddings.py")
    pdf.bullet("Flask sends PPE and detection events to the backend API")
    pdf.bullet("Backend stores records in Supabase and can trigger email/SMS alerts")

    pdf.section_title("3. Employee Enrollment (Face Recognition Setup)")
    pdf.body_text("When an admin adds an employee with profile photos:")
    pdf.bullet("Photos are saved to backend/uploads/{employee_name}/")
    pdf.bullet("Backend runs createEmbeddings.py on that folder")
    pdf.bullet("MTCNN detects the face in each photo")
    pdf.bullet("FaceNet (InceptionResnetV1, pretrained on VGGFace2) converts each face into a 512-dimensional embedding")
    pdf.bullet("Embeddings are saved as embeddings.csv in the employee folder")
    pdf.ln(2)
    pdf.body_text(
        "At Flask server startup, videoServer.py loads all employee embeddings and builds a "
        "Nearest Neighbors index for fast matching during live detection."
    )

    pdf.section_title("4. Face Recognition (Who Is This?)")
    pdf.body_text("On each processed frame or uploaded image:")
    pdf.bullet("MTCNN detects faces in the image")
    pdf.bullet("Each face is cropped and resized to 160x160 pixels")
    pdf.bullet("FaceNet generates a numeric embedding (face fingerprint)")
    pdf.bullet("The embedding is compared to stored employee embeddings using cosine similarity")
    pdf.bullet("If similarity >= 0.65, a match is found; otherwise the person is labeled Unknown")
    pdf.ln(2)
    pdf.subsection_title("Anti-Spoofing")
    pdf.body_text(
        "Each detected face is also checked using MiniFASNet (AntiSpoofPredict). "
        "If a photo or screen is detected instead of a live face, the label becomes "
        "\"Spoof Detected\" and the person is not treated as a valid recognition."
    )

    pdf.section_title("5. PPE Kit Detection")
    pdf.body_text(
        "PPE detection uses a YOLO object detection model. The system first tries to load a "
        "custom trained model from runs/detect/ppe_detection/weights/best.pt. If that is not "
        "available, it falls back to pretrained YOLOv12, YOLOv11, or YOLOv8 weights."
    )

    pdf.subsection_title("Required PPE Items")
    w2 = [35, 55, 80]
    pdf.table_row(["Item", "Also Detects As", "Threshold"], w2, bold=True)
    pdf.table_row(["Helmet", "hard-hat, safety-helmet", "0.75"], w2)
    pdf.table_row(["Gloves", "glove, safety-gloves", "0.65"], w2)
    pdf.table_row(["Boots", "boot, safety-shoes", "0.65"], w2)
    pdf.table_row(["Jacket", "vest, safety-vest, reflective-vest", "0.65"], w2)
    pdf.ln(4)

    pdf.subsection_title("Detection Pipeline")
    pdf.bullet("YOLO runs inference on the frame with initial confidence >= 0.5")
    pdf.bullet("Detections are filtered by per-class confidence thresholds")
    pdf.bullet("Spatial validation ensures PPE is near a detected Person bounding box")
    pdf.bullet("Helmets must appear in the upper portion of the person's body")
    pdf.bullet("check_ppe_compliance() marks a person compliant only if all 4 items are detected")
    pdf.ln(2)
    pdf.body_text(
        "For live video, spatial rules are strict to reduce false positives. For uploaded images, "
        "rules are more lenient because a full person box may not always be detected."
    )

    pdf.section_title("6. Live Camera Monitoring")
    pdf.body_text(
        "The Camera Monitor page displays a live MJPEG stream from Flask at /video_feed. "
        "The camera can be an RTSP IP security camera or a laptop/webcam."
    )
    pdf.subsection_title("Processing Loop")
    pdf.bullet("Every frame is streamed to the browser for live viewing")
    pdf.bullet("Every 8th frame is queued for background ML processing")
    pdf.bullet("Face recognition and PPE detection run in a background thread")
    pdf.bullet("Annotated bounding boxes are drawn on the stream (green = recognized, red = unknown, orange = spoof)")
    pdf.bullet("Flask POSTs to /api/ppe/event once per person per session to update compliance stats")
    pdf.bullet("Flask POSTs to /api/detections/event every 2 seconds per person for Recent Detections")
    pdf.bullet("Snapshot images are saved and shown in the frontend detection history")

    pdf.section_title("7. Image Analysis (Upload a Photo)")
    pdf.body_text(
        "The Image Analysis page uploads images to Flask /api/analyze-image. "
        "For images with multiple people, the system:"
    )
    pdf.bullet("Detects all faces in the image")
    pdf.bullet("Runs PPE detection on the whole image")
    pdf.bullet("Matches each PPE item to the nearest person using spatial proximity")
    pdf.bullet("Helmet must be near the face; gloves near hands; boots near feet; jacket near torso")
    pdf.ln(2)
    pdf.body_text("Example response per person:")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Courier", "", 8)
    pdf.multi_cell(pdf.epw, 4.5, (
        '{\n'
        '  "name": "John Doe",\n'
        '  "confidence": 0.82,\n'
        '  "ppe_compliant": false,\n'
        '  "ppe_items": { "helmet": true, "gloves": false, "boots": true, "jacket": true }\n'
        '}'
    ))
    pdf.ln(4)

    pdf.section_title("8. Backend Data Storage")
    pdf.body_text("When Flask reports a detection, the backend (ppeService.js) updates Supabase:")
    w3 = [50, 120]
    pdf.table_row(["Table / Data", "What Is Stored"], w3, bold=True)
    pdf.table_row(["workers", "Daily/total PPE violations, compliance streaks"], w3)
    pdf.table_row(["detection_events", "Who was seen, when, PPE status, camera source, snapshot path"], w3)
    pdf.table_row(["worker_details", "Employee name to worker ID mapping"], w3)
    pdf.ln(4)
    pdf.body_text(
        "The backend can also trigger email and SMS alerts (notificationService.js) when PPE "
        "violations are detected, listing which items are missing."
    )

    pdf.section_title("9. End-to-End Example")
    pdf.body_text("Scenario: Worker Raj walks past the site camera without gloves.")
    pdf.bullet("Camera captures frame; Flask processes it in the background")
    pdf.bullet("MTCNN finds Raj's face; FaceNet embedding matches Raj (confidence 0.78)")
    pdf.bullet("Anti-spoof check passes (real face, not a photo)")
    pdf.bullet("YOLO detects: helmet yes, gloves no, boots yes, jacket yes")
    pdf.bullet("check_ppe_compliance() returns non-compliant (gloves missing)")
    pdf.bullet("Flask draws annotations on the live stream")
    pdf.bullet("Flask POSTs to backend; Supabase records the violation")
    pdf.bullet("Frontend Recent Detections panel updates with Raj's snapshot")
    pdf.bullet("Admin can send email/SMS alert from the Camera Monitor UI")

    pdf.section_title("10. Key Source Files")
    w4 = [65, 105]
    pdf.table_row(["File", "Purpose"], w4, bold=True)
    pdf.table_row(["flaskServer/videoServer.py", "Main ML server: camera stream, image analysis"], w4)
    pdf.table_row(["flaskServer/ppeDetection.py", "YOLO PPE detection logic"], w4)
    pdf.table_row(["flaskServer/createEmbeddings.py", "Generates face embeddings for new employees"], w4)
    pdf.table_row(["backend/app.js", "API routes, employee management"], w4)
    pdf.table_row(["backend/ppeService.js", "PPE stats and violation tracking"], w4)
    pdf.table_row(["frontend/src/Pages/CameraMonitor.jsx", "Live camera UI"], w4)
    pdf.table_row(["frontend/src/Pages/ImageAnalysis.jsx", "Upload and analyze images"], w4)
    pdf.ln(4)

    pdf.section_title("11. Summary")
    w5 = [40, 55, 75]
    pdf.table_row(["Capability", "Technology", "How It Works"], w5, bold=True)
    pdf.table_row(["Who", "MTCNN + FaceNet", "Face to embedding to nearest match vs enrolled employees"], w5)
    pdf.table_row(["Spoof check", "MiniFASNet", "Rejects photos/screens of faces"], w5)
    pdf.table_row(["PPE kit", "YOLO (custom or pretrained)", "Detects helmet, gloves, boots, jacket with spatial rules"], w5)
    pdf.table_row(["Live monitoring", "Flask MJPEG + threads", "Processes every 8th frame, streams annotated video"], w5)
    pdf.table_row(["Records", "Node.js + Supabase", "Stores detections, violations, sends alerts"], w5)

    pdf.output(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_pdf()
    print(f"PDF created: {path}")
