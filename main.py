import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, Scale
from PIL import Image, ImageTk
import threading

# Initialize a stack to keep track of image states for undo functionality
image_stack = []

# Global variable to store the camera instance
camera = None

# Function to start camera and capture image
def start_camera():
    global camera
    
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            messagebox.showerror("Error", "Failed to open camera.")
            return
        
        # Start a thread to continuously read and display frames from the camera
        thread = threading.Thread(target=update_camera)
        thread.daemon = True  # Daemonize thread to close it when the main loop exits
        thread.start()
    else:
        messagebox.showwarning("Warning", "Camera is already running.")

# Function to continuously update camera preview
def update_camera():
    global camera
    
    if camera.isOpened():
        ret, frame = camera.read()
        if ret:
            # Convert the OpenCV BGR frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize the frame to fit the Tkinter window
            frame_resized = cv2.resize(frame_rgb, (800, 600))
            # Convert to ImageTk format
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            # Update the label with the new frame
            label_camera.config(image=img_tk)
            label_camera.image = img_tk
        # Repeat every 20 milliseconds
        root.after(20, update_camera)
    else:
        messagebox.showerror("Error", "Camera disconnected unexpectedly.")
        camera.release()
        label_camera.config(image="")  # Clear the camera preview

# Function to stop the camera
def stop_camera():
    global camera
    
    if camera is not None and camera.isOpened():
        camera.release()
        label_camera.config(image="")
        messagebox.showinfo("Info", "Camera stopped successfully.")
    else:
        messagebox.showwarning("Warning", "Camera is not running.")

# Function to save the captured image
def save_image():
    global camera
    
    if camera is not None and camera.isOpened():
        ret, frame = camera.read()
        if ret:
            # Ask user for file path to save the image
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                     filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                messagebox.showinfo("Success", "Image saved successfully.")
        else:
            messagebox.showwarning("Warning", "Failed to capture image from camera.")
    else:
        messagebox.showwarning("Warning", "Camera is not running.")

# Function to load an image
def load_image():
    global img, img_display, image_path, original_img
    image_path = filedialog.askopenfilename()
    if image_path:
        img = cv2.imread(image_path)
        if img is not None:
            # Clear the stack and add the initial image
            image_stack.clear()
            image_stack.append(img.copy())
            original_img = img.copy()
            display_image(img)
            adjust_window_size(img.shape[1], img.shape[0])
        else:
            messagebox.showerror("Error", "Could not open or find the image.")
    else:
        messagebox.showwarning("Warning", "No file selected.")

# Function to adjust Tkinter window size based on image dimensions
def adjust_window_size(img_width, img_height):
    root.geometry(f"{img_width + 200}x{img_height + 200}")

# Function to display the image in the Tkinter window
def display_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    img_display.config(image=img_tk)
    img_display.image = img_tk

# Function to resize the image
def resize_image():
    global img
    try:
        width = int(entry_width.get())
        height = int(entry_height.get())
        img_resized = cv2.resize(img, (width, height))
        image_stack.append(img_resized.copy())
        display_image(img_resized)
        img = img_resized
        adjust_window_size(width, height)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid dimensions.")

# Function to change the resolution
def change_resolution():
    global img
    try:
        resolution = int(entry_resolution.get())
        width = int(img.shape[1] * resolution / 100)
        height = int(img.shape[0] * resolution / 100)
        img_resized = cv2.resize(img, (width, height))
        image_stack.append(img_resized.copy())
        display_image(img_resized)
        img = img_resized
        adjust_window_size(width, height)
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid resolution percentage.")

# Function to apply grayscale filter
def apply_grayscale():
    global img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_display_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)  # To display in Tkinter
    image_stack.append(img_display_gray.copy())
    display_image(img_display_gray)
    img = img_gray

# Function to apply Gaussian blur filter
def apply_blur():
    global img
    img_blur = cv2.GaussianBlur(img, (15, 15), 0)
    image_stack.append(img_blur.copy())
    display_image(img_blur)
    img = img_blur

# Function to apply edge detection
def apply_edge_detection():
    global img
    img_edges = cv2.Canny(img, 100, 200)
    img_display_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)  # To display in Tkinter
    image_stack.append(img_display_edges.copy())
    display_image(img_display_edges)
    img = img_edges

# Function to adjust brightness and contrast
def adjust_brightness_contrast(brightness=0, contrast=100):
    global img, original_img
    img = original_img.copy()
    beta = brightness
    alpha = contrast / 100.0
    img_adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    display_image(img_adjusted)

# Function to rotate the image
def rotate_image():
    global img
    try:
        angle = int(entry_rotate.get())
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rotated = cv2.warpAffine(img, M, (w, h))
        image_stack.append(img_rotated.copy())
        display_image(img_rotated)
        img = img_rotated
        adjust_window_size(img.shape[1], img.shape[0])
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid rotation angle.")

# Function to flip the image
def flip_image():
    global img
    img_flipped = cv2.flip(img, 1)  # Flip horizontally
    image_stack.append(img_flipped.copy())
    display_image(img_flipped)
    img = img_flipped

# Function to invert colors
def invert_colors():
    global img
    img_inverted = cv2.bitwise_not(img)
    image_stack.append(img_inverted.copy())
    display_image(img_inverted)
    img = img_inverted

# Function to apply histogram equalization
def equalize_histogram():
    global img
    if len(img.shape) == 2:
        img_eq = cv2.equalizeHist(img)
    else:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    image_stack.append(img_eq.copy())
    display_image(img_eq)
    img = img_eq

# Function to apply dilation
def apply_dilation():
    global img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_dilated = cv2.dilate(img, kernel, iterations=1)
    image_stack.append(img_dilated.copy())
    display_image(img_dilated)
    img = img_dilated

# Function to apply erosion
def apply_erosion():
    global img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_eroded = cv2.erode(img, kernel, iterations=1)
    image_stack.append(img_eroded.copy())
    display_image(img_eroded)
    img = img_eroded

# Function to detect faces
def detect_faces():
    global img
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    img_faces = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)
    image_stack.append(img_faces.copy())
    display_image(img_faces)
    img = img_faces

# Function to undo the last operation
def undo():
    global img
    if len(image_stack) > 1:
        image_stack.pop()  # Remove the current state
        img = image_stack[-1].copy()  # Get the previous state
        display_image(img)
        adjust_window_size(img.shape[1], img.shape[0])
    else:
        messagebox.showwarning("Warning", "No more states to undo.")

# Function to update brightness and contrast using sliders
def update_brightness_contrast(val):
    brightness = slider_brightness.get()
    contrast = slider_contrast.get()
    adjust_brightness_contrast(brightness, contrast)

# Initialize the Tkinter window
root = tk.Tk()
root.title("Advanced Image Editor")

# Set initial window size
root.geometry("800x600")

# Create a canvas with scrollbars
canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a scrollbar to the canvas
scrollbar = tk.Scrollbar(root, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas to use the scrollbar
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create a frame to contain the widgets
frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

# Create and place the widgets
btn_load = tk.Button(frame, text="Load Image", command=load_image)
btn_load.pack(pady=10)

img_display = tk.Label(frame, bd=10, relief="groove")
img_display.pack(pady=10)

frame_resize = tk.Frame(frame, bg='#f0f0f0')
frame_resize.pack(pady=10)

tk.Label(frame_resize, text="Width:", bg='#f0f0f0').pack(side=tk.LEFT)
entry_width = tk.Entry(frame_resize, width=5)
entry_width.pack(side=tk.LEFT, padx=5)

tk.Label(frame_resize, text="Height:", bg='#f0f0f0').pack(side=tk.LEFT)
entry_height = tk.Entry(frame_resize, width=5)
entry_height.pack(side=tk.LEFT, padx=5)

btn_resize = tk.Button(frame_resize, text="Resize Image", command=resize_image)
btn_resize.pack(side=tk.LEFT, padx=5)

frame_resolution = tk.Frame(frame, bg='#f0f0f0')
frame_resolution.pack(pady=10)

tk.Label(frame_resolution, text="Resolution %:", bg='#f0f0f0').pack(side=tk.LEFT)
entry_resolution = tk.Entry(frame_resolution, width=5)
entry_resolution.pack(side=tk.LEFT, padx=5)

btn_resolution = tk.Button(frame_resolution, text="Change Resolution", command=change_resolution)
btn_resolution.pack(side=tk.LEFT, padx=5)

frame_filters = tk.Frame(frame, bg='#f0f0f0')
frame_filters.pack(pady=10)

btn_grayscale = tk.Button(frame_filters, text="Grayscale", command=apply_grayscale)
btn_grayscale.pack(side=tk.LEFT, padx=5)

btn_blur = tk.Button(frame_filters, text="Blur", command=apply_blur)
btn_blur.pack(side=tk.LEFT, padx=5)

btn_edges = tk.Button(frame_filters, text="Edge Detection", command=apply_edge_detection)
btn_edges.pack(side=tk.LEFT, padx=5)

btn_invert = tk.Button(frame_filters, text="Invert Colors", command=invert_colors)
btn_invert.pack(side=tk.LEFT, padx=5)

btn_equalize = tk.Button(frame_filters, text="Equalize Histogram", command=equalize_histogram)
btn_equalize.pack(side=tk.LEFT, padx=5)

btn_dilate = tk.Button(frame_filters, text="Dilation", command=apply_dilation)
btn_dilate.pack(side=tk.LEFT, padx=5)

btn_erode = tk.Button(frame_filters, text="Erosion", command=apply_erosion)
btn_erode.pack(side=tk.LEFT, padx=5)

btn_faces = tk.Button(frame_filters, text="Detect Faces", command=detect_faces)
btn_faces.pack(side=tk.LEFT, padx=5)

frame_adjustments = tk.Frame(frame, bg='#f0f0f0')
frame_adjustments.pack(pady=10)

slider_brightness = Scale(frame_adjustments, from_=-100, to=100, orient=tk.HORIZONTAL, label="Brightness", command=update_brightness_contrast)
slider_brightness.pack(padx=10, pady=5)

slider_contrast = Scale(frame_adjustments, from_=0, to=200, orient=tk.HORIZONTAL, label="Contrast", command=update_brightness_contrast)
slider_contrast.pack(padx=10, pady=5)

frame_rotate = tk.Frame(frame, bg='#f0f0f0')
frame_rotate.pack(pady=10)

tk.Label(frame_rotate, text="Rotate (degrees):", bg='#f0f0f0').pack(side=tk.LEFT)
entry_rotate = tk.Entry(frame_rotate, width=5)
entry_rotate.pack(side=tk.LEFT, padx=5)

btn_rotate = tk.Button(frame_rotate, text="Rotate Image", command=rotate_image)
btn_rotate.pack(side=tk.LEFT, padx=5)

btn_flip = tk.Button(frame, text="Flip Image", command=flip_image)
btn_flip.pack(pady=10)

btn_undo = tk.Button(frame, text="Undo", command=undo)
btn_undo.pack(pady=5)

btn_save = tk.Button(frame, text="Save Image", command=save_image)
btn_save.pack(pady=10)

# Button to open camera and capture image
btn_camera = tk.Button(frame, text="Open Camera & Capture", command=start_camera)
btn_camera.pack(pady=10)

# Create a label to display the camera preview
label_camera = tk.Label(root)
label_camera.pack()

# Run the Tkinter event loop
root.mainloop()


