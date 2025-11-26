import os
import cv2
import numpy as np
import pandas as pd
import sys
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import normalize

# Initialize models (same as faceRecognition.py)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def generate_embeddings(person_folder):
    embeddings = []
    image_names = []

    for file_name in os.listdir(person_folder):
        file_path = os.path.join(person_folder, file_name)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)
            if image is None:
                print(f"[WARNING] Could not read image: {file_name}")
                continue
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face using MTCNN
            with torch.no_grad():
                boxes, _ = mtcnn.detect(rgb_image)
                if boxes is None or len(boxes) == 0:
                    print(f"[WARNING] No face detected in: {file_name}")
                    continue
                
                # Use the first detected face
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box)
                face = rgb_image[y1:y2, x1:x2]
                
                if face.size == 0:
                    print(f"[WARNING] Empty face crop in: {file_name}")
                    continue
                
                # Resize to 160x160 (same as faceRecognition.py)
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0).to(device)
                
                # Generate embedding
                encoding = resnet(face_tensor).cpu().detach().numpy().flatten()
                embedding = normalize([encoding])[0]
                embeddings.append(embedding)
                image_names.append(file_name)
                print(f"[SUCCESS] Generated embedding for: {file_name}")

    if embeddings:
        df = pd.DataFrame(embeddings)
        csv_path = os.path.join(person_folder, 'embeddings.csv')
        df.to_csv(csv_path, index=False, header=False)
        print(f"[SUCCESS] Embeddings saved to {csv_path} ({len(embeddings)} embeddings)")
    else:
        print("[ERROR] No embeddings were generated.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Folder path not provided.")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    print(f"[INFO] Generating embeddings for: {folder_path}")
    generate_embeddings(folder_path)