import mtcnn
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import mediapipe as mp
import cv2
import os.path


class FaceCropper:

    def __init__(self, image_path, save_crop="face.jpg", save_mesh="face_mesh.jpg"):
        self.image_path = image_path
        self.save_crop = save_crop
        self.save_mesh = save_mesh

    def extract_face_from_image(self, required_size=(224, 224)):
        # load image and detect faces
        image = plt.imread(self.image_path)
        detector = mtcnn.MTCNN()
        faces = detector.detect_faces(image)
        face_images = []
        for face in faces:
            # extract the bounding box 50% more from the requested face
            x, y, w, h = face['box']
            b = max(0, y - (h // 2))
            d = min(image.shape[0], (y + h) + (h // 2))
            a = max(0, x - (w//2))
            c = min(image.shape[1], (x+w) + (w//2))
            face_boundary = image[b:d, a:c, :]
            # resize pixels to the model size
            face_image = Image.fromarray(face_boundary)
            face_array = asarray(face_image)
            face_images.append(face_array)
            plt.imsave(self.save_crop, face_images[0])
        return face_images

    def get_face_landmarks(self):
        mp_face_mesh = mp.solutions.face_mesh
        if not os.path.isfile("face.jpg"):
            self.extract_face_from_image()
        image_files = [self.save_crop]
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5) as face_mesh:
            for idx, file in enumerate(image_files):
                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # Print and draw face mesh landmarks on the image.
                if not results.multi_face_landmarks:
                    continue
                landmarks = []
                for data_point in results.multi_face_landmarks[0].landmark:
                    landmarks.append({
                        'X': data_point.x,
                        'Y': data_point.y,
                        'Z': data_point.z
                    })
        return landmarks

    def draw_face_mesh(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        if not os.path.isfile("face.jpg"):
            self.extract_face_from_image()
        image_files = [self.save_crop]
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5) as face_mesh:
            for idx, file in enumerate(image_files):
                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # Print and draw face mesh landmarks on the image.
                if not results.multi_face_landmarks:
                    continue
                annotated_image = image.copy()
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image=annotated_image, landmark_list=face_landmarks,
                                              connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        return annotated_image