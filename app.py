import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from tkinter import Label, Button, Frame
from collections import deque
from PIL import Image, ImageTk

class VirtualTryOnApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vashions")
        self.root.iconbitmap('static/vashions_icon.ico')  # Replace 'path_to_icon.ico' with the path to your .ico file
        self.current_design = None
        self.create_widgets()
        self.face_bbox_buffer = deque(maxlen=10)
    

    def create_widgets(self):
        # Create a main frame to hold the left and right frames
            self.main_frame = tk.Frame(self.root, width=1290, height=720)
            self.main_frame.pack()

            # Create the left frame
            self.left_frame = tk.Frame(self.main_frame, width=400, height=720, bg="black")
            self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
            self.left_frame.pack_propagate(False)  # Prevent resizing

            # Heading
            self.heading = ttk.Label(self.left_frame, text="Virtual Try On", font=("Helvetica", 24),background="black", foreground="lightgray")
            self.heading.pack(pady=20)

            # Necklace button
            self.necklace_button = ttk.Button(self.left_frame, text="Necklace", command=lambda: self.show_necklaces("Necklace"))
            self.necklace_button.pack(pady=10)

            # Ring button
            self.ring_button = ttk.Button(self.left_frame, text="Ring", command=lambda: self.show_rings("Ring"))
            self.ring_button.pack(pady=10)

            # Earring button
            self.earring_button = ttk.Button(self.left_frame, text="Earring", command=lambda: self.show_earring("Earring"))
            self.earring_button.pack(pady=10)

            # Bangle button
            self.bangle_button = ttk.Button(self.left_frame, text="Bangle", command=lambda: self.show_bangle("Bangle"))
            self.bangle_button.pack(pady=10)

            # Create the right frame
            self.right_frame = tk.Frame(self.main_frame, width=890, height=720, bg="white")
            self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            # Load the image
            image_path = 'static/profile.jpg'
            image = Image.open(image_path)
            resized_image = image.resize((890, 720))  # Resize the image to fit the frame
            photo = ImageTk.PhotoImage(resized_image)

            # Create a label with the image and add it to the right frame
            label = tk.Label(self.right_frame, image=photo)
            label.image = photo  # Keep a reference to prevent garbage collection
            label.pack(fill=tk.BOTH, expand=True)

            # Webcam display area on the right frame
            self.webcam_canvas = tk.Canvas(self.right_frame, width=890, height=720, bg="white")
            self.webcam_canvas.pack(fill=tk.BOTH, expand=True)

            # Initialize MediaPipe Face Detection.
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

            # Initialize MediaPipe Hands.
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh()

            # Default necklace path
            self.necklace_image_path = 'static/Image/Necklace/necklace_1.png'
            self.necklace_image = cv2.imread(self.necklace_image_path, cv2.IMREAD_UNCHANGED)
            
            # Default ring path
            self.ring_image_path = 'static/Image/Ring/ring_1.png'
            self.ring_image = cv2.imread(self.ring_image_path, cv2.IMREAD_UNCHANGED)
            
            # Default earring image path
            self.earring_image_path = 'static/Image/Earring/earring_1.png'
            self.earring_image = cv2.imread(self.earring_image_path, cv2.IMREAD_UNCHANGED)

            # Default bangle image path
            self.bangle_image_path = 'static/Image/Bangle/bangle_1.png'
            self.bangle_image = cv2.imread(self.bangle_image_path, cv2.IMREAD_UNCHANGED)
           

    def show_necklaces(self, design):
        self.current_design = design
        self.show_virtual_try_on([r"static/Image/Necklace/necklace_1.png", 
                                  r"static/Image/Necklace/necklace_20.png",
                                  r"static/Image/Necklace/necklace_24.png",
                                  r"static/Image/Necklace/necklace_28.png",
                                  ],design)

    def show_rings(self, design):
        self.current_design = design
        self.show_virtual_try_on([r"static/Image/Ring/ring_1.png", 
                                  r"static/Image/Ring/ring_2.png",
                                  r"static/Image/Ring/ring_3.png",
                                  r"static/Image/Ring/ring_4.png",
                                  r"static/Image/Ring/ring_5.png"
                                  ],design)
    
    def show_earring(self, design):
        self.current_design = design
        self.show_virtual_try_on([r"static/Image/Earring/earring_1.png", 
                                  r"static/Image/Earring/earring_2.png",
                                  r"static/Image/Earring/earring_3.png",
                                  r"static/Image/Earring/earring_4.png",
                                  r"static/Image/Earring/earring_5.png"],design)
    
    def show_bangle(self, design):
        self.current_design = design
        self.show_virtual_try_on([r"static/Image/Bangle/bangle_1.png",
                                  r"static/Image/Bangle/bangle_2.png",
                                  r"static/Image/Bangle/bangle_3.png",
                                  r"static/Image/Bangle/bangle_4.png"],design)

    def change_necklace_image(self, path):
        self.necklace_image_path = path
        self.necklace_image = cv2.imread(self.necklace_image_path, cv2.IMREAD_UNCHANGED)
        print(f"Changed necklace image to {self.necklace_image_path}")

    def change_ring_image(self, path):
        self.ring_image_path = path
        self.ring_image = cv2.imread(self.ring_image_path, cv2.IMREAD_UNCHANGED)
        print(f"Changed ring image to {self.ring_image_path}")

    def change_earring_image(self, path):
        self.earring_image_path = path
        self.earring_image = cv2.imread(self.earring_image_path, cv2.IMREAD_UNCHANGED)
        print(f"Changed earring image to {self.earring_image_path}")

    def change_bangle_image(self, path):
        self.bangle_image_path = path
        self.bangle_image = cv2.imread(self.bangle_image_path, cv2.IMREAD_UNCHANGED)
        print(f"Changed bangle image to {self.bangle_image_path}")


    def show_virtual_try_on(self, image_list, design):
        self.try_on_window = tk.Toplevel(self.root)
        self.try_on_window.title("Virtual Try On")
        self.try_on_window.is_fullscreen = False
        # Main frame to hold left and right frames
        main_frame = tk.Frame(self.try_on_window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame for webcam and buttons
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right frame for image list and scrollbar
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Webcam canvas
        self.webcam_canvas = tk.Canvas(left_frame, width=650, height=650)
        self.webcam_canvas.pack(side=tk.TOP, pady=10)

        # Frame for buttons below the webcam canvas
        button_frame = tk.Frame(left_frame)
        button_frame.pack(side=tk.TOP, pady=10)

        self.quit_icon = ImageTk.PhotoImage(Image.open("static/icons/quit_icon.png").resize((20, 20), Image.LANCZOS))
        self.capture_icon = ImageTk.PhotoImage(Image.open("static/icons/capture_icon.png").resize((20, 20), Image.LANCZOS))
        self.fullscreen_icon = ImageTk.PhotoImage(Image.open("static/icons/fullscreen_icon.png").resize((20, 20), Image.LANCZOS))

        # Add three buttons to the button frame
        button1 = tk.Button(button_frame, command=self.button1_action,image=self.quit_icon)
        button1.pack(side=tk.LEFT)

        button3 = tk.Button(button_frame, command=self.button3_action,image=self.capture_icon)
        button3.pack(side=tk.LEFT)

        button2 = tk.Button(button_frame,command=self.button2_action,image=self.fullscreen_icon)
        button2.pack(side=tk.LEFT)

        # Create a scrollbar for the image list frame
        scrollbar = tk.Scrollbar(right_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create a canvas for the image list frame
        canvas = tk.Canvas(right_frame, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure the scrollbar to scroll the canvas
        scrollbar.config(command=canvas.yview)

        # Create a frame inside the canvas to hold the images
        image_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=image_frame, anchor=tk.NW)

        # Function to update the scroll region
        def update_scroll_region(event):
            canvas.config(scrollregion=canvas.bbox(tk.ALL))

        # Bind the update_scroll_region function to the frame resize event
        image_frame.bind("<Configure>", update_scroll_region)

        # Populate the image frame with images
        for idx, image_path in enumerate(image_list, start=1):
            # Item number label
            item_label = tk.Label(image_frame, text=f"{design} {idx}")
            item_label.grid(row=idx-1, column=0, pady=(10, 0), padx=10)  # item number above the image

            # Image
            image = Image.open(image_path)
            image = image.resize((150, 150), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            # Create a label for each image
            image_label = tk.Label(image_frame, image=photo)
            image_label.image = photo  # keep a reference!
            image_label.image_path = image_path  # store path in label
            image_label.grid(row=idx-1, column=1, pady=(10, 0), padx=10)  # image label next to item number
            image_label.bind("<Button-1>", self.on_image_click)  # bind click event

        # Start webcam
        self.cap = cv2.VideoCapture(0)
        self.update_frame()


    def button1_action(self):
        # Define the action for button 1
        self.cap.release()
        self.try_on_window.destroy()

    def button2_action(self):
        # Define the action for button 2
        self.try_on_window.is_fullscreen = not self.try_on_window.is_fullscreen
        self.try_on_window.attributes("-fullscreen", self.try_on_window.is_fullscreen)

    def button3_action(self):
    # Ensure the frame with jewelry overlay is captured
        ret, frame = self.cap.read()
        if ret:
            # Apply the jewelry overlay to the frame
            frame_with_overlay = self.apply_jewelry_overlay(frame)

            # Convert the frame to PIL image format
            pil_image = Image.fromarray(cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB))

            # Save the PIL image
            pil_image.save("captured_image_with_jewelry.png")
            print("Image captured and saved as captured_image_with_jewelry.png")
        def apply_jewelry_overlay(self, frame):
            # Convert the center section image from OpenCV BGR format to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Check the current jewelry type and apply the appropriate overlay
            if self.current_design == "Necklace":
                frame_with_overlay = self.add_necklace_overlay(frame_rgb)
            elif self.current_design == "Ring":
                frame_with_overlay = self.add_ring_overlay(frame_rgb)
            elif self.current_design == "Earring":
                frame_with_overlay = self.add_earring_overlay(frame_rgb)
            elif self.current_design == "Bangle":
                frame_with_overlay = self.add_bangle_overlay(frame_rgb)

            return frame_with_overlay


        
    def on_image_click(self, event):
        image_path = event.widget.image_path
        if self.current_design == "Necklace":
            self.change_necklace_image(image_path)
        elif self.current_design == "Ring":
            self.change_ring_image(image_path)
        elif self.current_design == "Earring":
            self.change_earring_image(image_path)
        elif self.current_design == "Bangle":
            self.change_bangle_image(image_path)
        # Add more conditions if you have other types like Earring, Bangle

    def update_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            height, width, _ = frame.shape

            # Calculate the width of each section
            center_width = int(width * 0.35)
            side_width = (width - center_width) // 2

            # Divide the frame into three sections
            left_section = frame[:, :side_width]
            center_section = frame[:, side_width:side_width + center_width]
            right_section = frame[:, side_width + center_width:]
            
            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the center section image from OpenCV BGR format to RGB
            frame_rgb = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)

            if self.current_design == "Necklace":
                # Perform face detection on the center section
                results = self.face_detection.process(frame_rgb)

                if results.detections:
                    for detection in results.detections:
                        # Extract the bounding box coordinates
                        bboxC = detection.location_data.relative_bounding_box
                        hC, wC, _ = center_section.shape
                        xminC = int(bboxC.xmin * wC)
                        yminC = int(bboxC.ymin * hC)
                        widthC = int(bboxC.width * wC)
                        heightC = int(bboxC.height * hC)
                        xmaxC = xminC + widthC
                        ymaxC = yminC + heightC

                        # Calculate the shoulder to chest region
                        shoulder_ymin = ymaxC 
                        chest_ymax = min(ymaxC + 115, hC)

                        # Adjust the width of the bounding box for the necklace
                        xminC -= -8  # Decrease the left side
                        xmaxC += 2.5  # Increase the right side

                        # Check if the bounding box dimensions are valid
                        if widthC > 0 and heightC > 0 and xmaxC > xminC and chest_ymax > shoulder_ymin:
                            # Add the current bounding box to the buffer
                            self.face_bbox_buffer.append((xminC, yminC, xmaxC, ymaxC, shoulder_ymin, chest_ymax))

                            # Calculate the average bounding box
                            avg_bbox = np.mean(self.face_bbox_buffer, axis=0).astype(int)
                            avg_xminC, avg_yminC, avg_xmaxC, avg_ymaxC, avg_shoulder_ymin, avg_chest_ymax = avg_bbox

                            # Resize necklace image to fit the bounding box size
                            resized_image = cv2.resize(self.necklace_image, (avg_xmaxC - avg_xminC, avg_chest_ymax - avg_shoulder_ymin))

                            # Calculate the start and end coordinates for the necklace image
                            start_x = avg_xminC
                            start_y = avg_shoulder_ymin
                            end_x = start_x + (avg_xmaxC - avg_xminC)
                            end_y = start_y + (avg_chest_ymax - avg_shoulder_ymin)

                            # Create a mask from the alpha channel
                            alpha_channel = resized_image[:, :, 3]
                            mask = alpha_channel[:, :, np.newaxis] / 255.0

                            # Apply the mask to the necklace image
                            overlay = resized_image[:, :, :3] * mask

                            # Create a mask for the input image region
                            mask_inv = 1 - mask

                            # Apply the inverse mask to the input image
                            region = center_section[start_y:end_y, start_x:end_x]
                            if region.shape[1] > 0 and region.shape[0] > 0:
                                resized_mask_inv = cv2.resize(mask_inv, (region.shape[1], region.shape[0]))
                                resized_mask_inv = resized_mask_inv[:, :, np.newaxis]  # Add an extra dimension to match the number of channels

                                region_inv = region * resized_mask_inv

                                # Combine the resized image and the input image region
                                if region_inv.shape[1] > 0 and region_inv.shape[0] > 0:
                                    resized_overlay = cv2.resize(overlay, (region_inv.shape[1], region_inv.shape[0]))
                                    region_combined = cv2.add(resized_overlay, region_inv)

                                    # Replace the neck region in the input image with the combined region
                                    center_section[start_y:end_y, start_x:end_x] = region_combined

                else:
                        # Display text message when no face is detected
                        if center_section is not None and center_section.shape[0] > 0 and center_section.shape[1] > 0:
                            # Define the text and its properties
                            text = "Readjust"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            color = (0, 0, 255)  # Red color in BGR
                            thickness = 2

                            # Get the text size
                            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                            # Calculate the text position
                            text_x = (center_section.shape[1] - text_size[0])//2;
                            text_y = (center_section.shape[0] + text_size[1])//2;

                            # Put the text on the image
                            cv2.putText(center_section, text, (text_x, text_y), font, font_scale, color, thickness)
                        else:
                            print("center_section is None or empty. Cannot display text.")
                                    
            elif self.current_design == "Ring":
                
                # Convert the center section image from OpenCV BGR format to RGB
                frame_rgb = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)
                
                # Perform hand detection on the center section
                hand_results = self.hands.process(frame_rgb)

                # Check if any hands were detected in the center section
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Iterate over the landmarks and draw them on the frame
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            cx, cy = int(landmark.x * center_width), int(landmark.y * height)

                            # Check if hand is within the valid region
                            if 0 <= cx <= center_width:
                                hand_in_frame = True

                                # Get the pixel coordinates of points 13 and 14 (ring_finger)
                                point13 = hand_landmarks.landmark[13]
                                point14 = hand_landmarks.landmark[14]
                                point13_x, point13_y = int(point13.x * center_width), int(point13.y * height)
                                point14_x, point14_y = int(point14.x * center_width), int(point14.y * height)

                                # Calculate the center coordinates and size of the bounding box for points 13 and 14
                                bbox_size = 20  # Adjust the size of the bounding box as needed
                                center_x = (point13_x + point14_x) // 2
                                center_y = (point13_y + point14_y) // 2
                                x1, y1 = center_x - bbox_size, center_y - bbox_size
                                x2, y2 = center_x + bbox_size, center_y + bbox_size

                                # Resize the ring image to fit the bounding box size
                                resized_ring = cv2.resize(self.ring_image, (x2 - x1, y2 - y1))

                                # Define the region of interest for placing the ring image
                                roi_ring = center_section[y1:y2, x1:x2]

                                # Create a mask from the alpha channel of the ring image
                                ring_alpha = resized_ring[:, :, 3] / 255.0
                                mask_ring = np.stack([ring_alpha] * 3, axis=2)

                                # Apply the mask to the ring image if the shapes match
                                if roi_ring.shape == mask_ring.shape:
                                    # Apply the mask to the ring image
                                    masked_ring = resized_ring[:, :, :3] * mask_ring

                                    # Create a mask for the region of interest of the ring image
                                    roi_mask_ring = 1 - mask_ring

                                    # Apply the inverse mask to the region of interest
                                    roi_combined_ring = roi_ring * roi_mask_ring

                                    # Combine the masked ring image and the region of interest
                                    combined_ring = cv2.add(masked_ring, roi_combined_ring)

                                    # Place the combined ring image back into the center section
                                    center_section[y1:y2, x1:x2] = combined_ring

                else:
                        # Display text message when no face is detected
                        if center_section is not None and center_section.shape[0] > 0 and center_section.shape[1] > 0:
                            # Define the text and its properties
                            text = "Not detectable"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            color = (0, 0, 255)  # Red color in BGR
                            thickness = 2

                            # Get the text size
                            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                            # Calculate the text position
                            text_x = (center_section.shape[1] - text_size[0])//2;
                            text_y = (center_section.shape[0] + text_size[1])//2;

                            # Put the text on the image
                            cv2.putText(center_section, text, (text_x, text_y), font, font_scale, color, thickness)
                        else:
                            print("center_section is None or empty. Cannot display text.")
                                
            
            elif self.current_design == "Earring":
                # Perform face mesh detection on the center section
                results = self.face_mesh.process(frame_rgb)

                # Index numbers for the left and right ear landmarks
                left_ear_index = 177
                right_ear_index = 401

                # Flags to check if both left and right earrings are detected
                left_ear_detected = False
                right_ear_detected = False

                if results.multi_face_landmarks:
                    for facial_landmarks in results.multi_face_landmarks:
                        # Calculate bounding box coordinates for left ear
                        left_ear_landmark = facial_landmarks.landmark[left_ear_index]
                        left_ear_x = int(left_ear_landmark.x * center_section.shape[1])
                        left_ear_y = int(left_ear_landmark.y * center_section.shape[0])
                        left_ear_bbox_size = 15
                        left_ear_top_left = (left_ear_x - 10 - left_ear_bbox_size, left_ear_y - left_ear_bbox_size)
                        left_ear_bottom_right = (left_ear_x - 10 + left_ear_bbox_size, left_ear_y + left_ear_bbox_size)

                        # Check if left ear is within the central_section
                        if left_ear_top_left[0] >= 0 and left_ear_top_left[1] >= 0 and \
                        left_ear_bottom_right[0] <= center_section.shape[1] and left_ear_bottom_right[1] <= center_section.shape[0]:
                            # Resize the earring image to match the size of the bounding box
                            ear_width = left_ear_bottom_right[0] - left_ear_top_left[0]
                            ear_height = left_ear_bottom_right[1] - left_ear_top_left[1]
                            resized_earring = cv2.resize(self.earring_image, (ear_width, ear_height))

                            # Convert earring image to a 3-channel image with alpha channel
                            resized_earring_rgb = cv2.cvtColor(resized_earring, cv2.COLOR_BGRA2BGR)

                            # Create a mask from the alpha channel
                            alpha_channel = resized_earring[:, :, 3]
                            mask = alpha_channel[:, :, np.newaxis] / 255.0

                            # Apply the mask to the resized earring image
                            overlay = resized_earring_rgb * mask

                            # Create a mask for the input image region
                            mask_inv = 1 - mask

                            # Apply the inverse mask to the input image for the left earring
                            region_left = center_section[left_ear_top_left[1]:left_ear_bottom_right[1],
                                        left_ear_top_left[0]:left_ear_bottom_right[0]]
                            region_left_inv = region_left * mask_inv

                            # Combine the resized earring image and the input image regions
                            region_left_combined = cv2.add(overlay, region_left_inv)

                            # Replace the left ear region in the input image with the combined region for the left earring
                            center_section[left_ear_top_left[1]:left_ear_bottom_right[1],
                                        left_ear_top_left[0]:left_ear_bottom_right[0]] = region_left_combined

                            left_ear_detected = True

                        # Calculate bounding box coordinates for right ear
                        right_ear_landmark = facial_landmarks.landmark[right_ear_index]
                        right_ear_x = int(right_ear_landmark.x * center_section.shape[1])
                        right_ear_y = int(right_ear_landmark.y * center_section.shape[0])
                        right_ear_bbox_size = 15
                        right_ear_top_left = (right_ear_x + 10 - right_ear_bbox_size, right_ear_y - right_ear_bbox_size)
                        right_ear_bottom_right = (right_ear_x + 10 + right_ear_bbox_size, right_ear_y + right_ear_bbox_size)

                        # Check if right ear is within the central_section
                        if right_ear_top_left[0] >= 0 and right_ear_top_left[1] >= 0 and \
                        right_ear_bottom_right[0] <= center_section.shape[1] and right_ear_bottom_right[1] <= center_section.shape[0]:
                            # Resize the earring image to match the size of the bounding box
                            ear_width = right_ear_bottom_right[0] - right_ear_top_left[0]
                            ear_height = right_ear_bottom_right[1] - right_ear_top_left[1]
                            resized_earring = cv2.resize(self.earring_image, (ear_width, ear_height))

                            # Convert earring image to a 3-channel image with alpha channel
                            resized_earring_rgb = cv2.cvtColor(resized_earring, cv2.COLOR_BGRA2BGR)

                            # Create a mask from the alpha channel
                            alpha_channel = resized_earring[:, :, 3]
                            mask = alpha_channel[:, :, np.newaxis] / 255.0

                            # Apply the mask to the resized earring image
                            overlay = resized_earring_rgb * mask

                            # Create a mask for the input image region
                            mask_inv = 1 - mask

                            # Apply the inverse mask to the input image for the right earring
                            region_right = center_section[right_ear_top_left[1]:right_ear_bottom_right[1],
                                                        right_ear_top_left[0]:right_ear_bottom_right[0]]
                            region_right_inv = region_right * mask_inv

                            # Combine the resized earring image and the input image regions
                            region_right_combined = cv2.add(overlay, region_right_inv)

                            # Replace the right ear region in the input image with the combined region for the right earring
                            center_section[right_ear_top_left[1]:right_ear_bottom_right[1],
                                        right_ear_top_left[0]:right_ear_bottom_right[0]] = region_right_combined

                            right_ear_detected = True

                    # If both left and right earrings are detected and displayed, set hand_in_frame to True
                    hand_in_frame = left_ear_detected and right_ear_detected

                else:
                        # Display text message when no face is detected
                        if center_section is not None and center_section.shape[0] > 0 and center_section.shape[1] > 0:
                            # Define the text and its properties
                            text = "Readjust"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            color = (0, 0, 255)  # Red color in BGR
                            thickness = 2

                            # Get the text size
                            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                            # Calculate the text position
                            text_x = (center_section.shape[1] - text_size[0])//2;
                            text_y = (center_section.shape[0] + text_size[1])//2;

                            # Put the text on the image
                            cv2.putText(center_section, text, (text_x, text_y), font, font_scale, color, thickness)
                        else:
                            print("center_section is None or empty. Cannot display text.")
                                    

            elif self.current_design == "Bangle":
                # Perform hand detection on the center section
                hand_results = self.hands.process(frame_rgb)

                # Check if any hands were detected in the center section
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Get the coordinates of the wrist (point 0)
                        wrist = hand_landmarks.landmark[0]
                        wrist_x = int(wrist.x * center_section.shape[1])
                        wrist_y = int(wrist.y * center_section.shape[0]) + 50  # Adjust the y-coordinate by adding the offset

                        # Define the bounding box parameters
                        box_width = 120
                        box_height = 50
                        half_width = box_width // 2
                        half_height = box_height // 2
                        top_left = (wrist_x - half_width, wrist_y - half_height)
                        bottom_right = (wrist_x + half_width, wrist_y + half_height)

                        # Check if the bounding box is within the frame
                        if top_left[0] >= 0 and top_left[1] >= 0 and bottom_right[0] < center_section.shape[1] and bottom_right[1] < center_section.shape[0]:
                            # Resize the bangle image to fit the bounding box size
                            resized_bangle = cv2.resize(self.bangle_image, (box_width, box_height))

                            # Define the region of interest for placing the bangle image
                            roi = center_section[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                            # Create a mask from the alpha channel of the bangle image
                            bangle_alpha = resized_bangle[:, :, 3] / 255.0
                            mask = np.stack([bangle_alpha] * 3, axis=2)

                            # Apply the mask to the bangle image
                            masked_bangle = resized_bangle[:, :, :3] * mask

                            # Create a mask for the region of interest
                            roi_mask = 1 - mask

                            # Apply the inverse mask to the region of interest
                            roi_combined = roi * roi_mask

                            # Combine the masked bangle image and the region of interest
                            combined = cv2.add(masked_bangle, roi_combined)

                            # Replace the region of interest with the combined image
                            center_section[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = combined
                
                else:
                        # Display text message when no face is detected
                        if center_section is not None and center_section.shape[0] > 0 and center_section.shape[1] > 0:
                            # Define the text and its properties
                            text = "Readjust"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            color = (0, 0, 255)  # Red color in BGR
                            thickness = 2

                            # Get the text size
                            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                            # Calculate the text position
                            text_x = (center_section.shape[1] - text_size[0])//2;
                            text_y = (center_section.shape[0] + text_size[1])//2;

                            # Put the text on the image
                            cv2.putText(center_section, text, (text_x, text_y), font, font_scale, color, thickness)
                        else:
                            print("center_section is None or empty. Cannot display text.")
                                    

            # Merge the sections back into the frame
            frame[:, :side_width] = left_section
            frame[:, side_width:side_width + center_width] = center_section
            frame[:, side_width + center_width:] = right_section
            

            frame = cv2.resize(frame, (650, 650))
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tk = ImageTk.PhotoImage(image=image)
            self.webcam_canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
            self.webcam_canvas.image_tk = image_tk

        self.root.after(10, self.update_frame)

# Create and run the application
root = tk.Tk()
app = VirtualTryOnApp(root)
root.mainloop()
