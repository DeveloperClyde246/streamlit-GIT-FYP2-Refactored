import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

class Preprocessor:
    def __init__(self):
        self.data = None

    def load_3000_data(self, data_path):
        # Get emotion categories (folder names)
        categories = sorted(os.listdir(data_path))  # Ensure consistent ordering
        print("Categories:", categories)

        # Map categories to numerical labels
        label_map = {category: idx for idx, category in enumerate(categories)}
        print("Label Map:", label_map)

        images = []
        labels = []
        
        for category, label in label_map.items():
            category_path = os.path.join(data_path, category)
            
            # Load all images in the category folder
            category_images = []
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                image = cv2.imread(file_path)
                category_images.append((image, label))
            
            # Randomly sample 3000 images for this category
            if len(category_images) > 3000:
                category_images = random.sample(category_images, 3000)
            
            for image, label in category_images:
                images.append(image)
                labels.append(label)

        self.data = images
        self.convert_to_grayscale()
        self.resize_image((48, 48))
        self.histogram_equalization()
        self.normalize()
        images = self.data
        
        # Convert to numpy arrays
        images = np.array(images, dtype='float32')
        labels = np.array(labels)
        
        return images, labels

    def load_1_data(self, data_path):
        #load one image from the selected path
        images = []
        labels = []
        image = cv2.imread(data_path)
        images.append(image)
        labels.append(0)

        self.data = images
        self.convert_to_grayscale()
        self.resize_image((48, 48))
        self.histogram_equalization()
        self.normalize()
        images = self.data
        
        # Convert to numpy arrays
        images = np.array(images, dtype='float32')
        labels = np.array(labels)
        
        return images, labels

    def load_data(self, data_path):
        # Get emotion categories (folder names)
        categories = sorted(os.listdir(data_path))  # Ensure consistent ordering
        print("Categories:", categories)

        # Map categories to numerical labels
        label_map = {category: idx for idx, category in enumerate(categories)}
        print("Label Map:", label_map)

        images = []
        labels = []
        
        for category, label in label_map.items():
            category_path = os.path.join(data_path, category)
            
            # Load all images in the category folder
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                image = cv2.imread(file_path)
                images.append(image)
                labels.append(label)

        self.data = images
        self.convert_to_grayscale()
        self.resize_image((48, 48))
        self.histogram_equalization()
        self.normalize()
        images = self.data
        
        # Convert to numpy arrays
        images = np.array(images, dtype='float32')
        labels = np.array(labels)
        return images, labels
    
    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        success = True
        count = 0
        while success:
            success, frame = cap.read()
            if not success or frame is None:
                print("Video fully processed. All frames extracted.")
                break
            frames.append(frame)
            count += 1
        cap.release()
        print(count)
        self.data = frames
        return self
    
    def detect_and_crop_face(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        cropped_faces = []
        for frame in self.data:
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                cropped_face = frame[y:y+h, x:x+w]
                cropped_faces.append(cropped_face)
        self.data = cropped_faces
        return self
    
    def detect_and_crop_left_eye(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        cropped_eyes = []
        for frame in self.data:
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi_gray = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
                if len(eyes) > 0:
                    # Sort eyes by x-coordinate to get the right eye
                    eyes = sorted(eyes, key=lambda e: e[0], reverse=True)
                    right_eye = eyes[0]
                    (ex, ey, ew, eh) = right_eye
                    cropped_eye = roi_gray[ey:ey+eh, ex:ex+ew]
                    cropped_eyes.append(cropped_eye)
        self.data = cropped_eyes
        return self
    
    def check_image_size(self):
        #show the max and min shape of the images
        max_shape = (0, 0)
        min_shape = (float('inf'), float('inf'))
        for image in self.data:
            max_shape = (max(max_shape[0], image.shape[0]), max(max_shape[1], image.shape[1]))
            min_shape = (min(min_shape[0], image.shape[0]), min(min_shape[1], image.shape[1]))
        print("Max Shape:", max_shape)
        print("Min Shape:", min_shape)
    
    def gaussian_smoothing(self):
        blurred_images = []
        for image in self.data:
            blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
            blurred_images.append(blurred_image)
        self.data = blurred_images
        return self

    def resize_image(self, target_size):
        resized_images = []
        for image in self.data:
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            resized_images.append(resized_image)
        self.data = resized_images
        return self

    def convert_to_grayscale(self):
        grayscale_images = []
        for image in self.data:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayscale_images.append(grayscale_image)
        self.data = grayscale_images
        return self

    def histogram_equalization(self):
        equalized_images = []
        for image in self.data:
            equalized_image = cv2.equalizeHist(image)
            equalized_images.append(equalized_image)
        self.data = equalized_images
        return self

    def normalize(self):
        normalized_images = []
        for image in self.data:
            normalized_image = image / 255.0
            normalized_images.append(normalized_image)
        self.data = normalized_images
        return self
    
    def show_images(self, num_images):
        num_images = max(1, num_images)  # Ensure at least 1 image is displayed
        num_rows = (num_images + 4) // 5  # Calculate the number of rows with max 5 images per row
        num_cols = min(num_images, 5)  # Ensure at most 5 images per row

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))  # Adjust height based on rows

        # Handle the case where only one image is shown
        if num_images == 1:
            axes = [axes]  # Convert a single Axes object to a list for uniform indexing
        else:
            axes = axes.flatten()  # Flatten the axes array for easy indexing

        # Display images in the plot
        for i in range(num_images):
            ax = axes[i]
            ax.imshow(cv2.cvtColor(self.data[i], cv2.COLOR_BGR2RGB))
            ax.axis('off')

        # Remove any unused subplots
        for i in range(num_images, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    def preprocess(self,video_path):
        self.load_video(video_path)
        self.convert_to_grayscale()
        self.detect_and_crop_face()
        self.resize_image((48, 48))
        self.histogram_equalization()
        self.normalize()
        return self.data
    
    def get_data(self):
        return self.data