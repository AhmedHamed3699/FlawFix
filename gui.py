import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageFilter

def upload_image():
    """Prompt the user to upload an image."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        # Open and display the uploaded image
        uploaded_img = Image.open(file_path)
        uploaded_img.thumbnail((300, 300))  # Resize for display purposes
        uploaded_img_tk = ImageTk.PhotoImage(uploaded_img)
        label_uploaded_img.config(image=uploaded_img_tk)
        label_uploaded_img.image = uploaded_img_tk
        
        # Save for processing
        global input_image
        input_image = uploaded_img

def process_image():
    """Call processing functions on the uploaded image."""
    if input_image:
        ###
        processed_img = input_image.filter(ImageFilter.BLUR)
        processed_img.thumbnail((300, 300))  # Resize for display purposes
        
        processed_img_tk = ImageTk.PhotoImage(processed_img)
        label_processed_img.config(image=processed_img_tk)
        label_processed_img.image = processed_img_tk
        
        # Save output image (optional)
        processed_img.save("output_image.png")

# Create the main window
root = tk.Tk()
root.title("Image Processing App")
root.geometry("1000x700")
root.configure(bg="#2c3e50")

# Header with title
header_frame = tk.Frame(root, bg="#34495e", pady=20)
header_frame.pack(fill="x")

header_label = tk.Label(header_frame, text="Flaw Fix", font=("Helvetica", 20, "bold"), fg="white", bg="#34495e")
header_label.pack()

# Upload Frame
upload_frame = tk.Frame(root, bg="#2c3e50", pady=20)
upload_frame.pack()

instructions_label = tk.Label(upload_frame, text="Upload an image to process:", font=("Arial", 14), fg="white", bg="#2c3e50")
instructions_label.pack(pady=10)

btn_upload = ttk.Button(upload_frame, text="Upload Image", command=upload_image)
btn_upload.pack(pady=10)

# Create a frame to hold the two images side by side
frame_images = tk.Frame(root, bg="#2c3e50")
frame_images.pack(pady=20)

# Create the uploaded image label and pack it to the left
label_uploaded_img = tk.Label(frame_images, bg="#2c3e50")
label_uploaded_img.pack(side="left", padx=10)  # Add padding between images

# Create the processed image label and pack it to the left (side by side with the first label)
label_processed_img = tk.Label(frame_images, bg="#2c3e50")
label_processed_img.pack(side="left", padx=10)  # Add padding between images

# Process Button
btn_process = ttk.Button(root, text="Process Image", command=process_image)
btn_process.pack(pady=20)

# Processed Image Display

# Styling with ttkbootstrap
try:
    import ttkbootstrap as ttkb
    style = ttkb.Style("superhero")  # Try other themes like 'darkly', 'flatly', etc.
    root.tk.call("source", ttkb.TTK_THEME_PATH)
except ImportError:
    pass

# Run the application
root.mainloop()