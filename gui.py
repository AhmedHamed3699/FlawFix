import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageFilter
import cv2

from core.enhancement.chromatic_aberration import CACorrection
from core.preprocessing.noise_reduction import apply_noise_removal
from core.preprocessing.brightness import adjust_brightness_to_range
from core.preprocessing.contrast import enhance_contrast
from core.restoration.scratch_removal import scratch_removal

input_image = None


def upload_image():
    """Prompt the user to upload an image."""
    after_img_label.config(image=None)
    after_img_label.image = None

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.*")])
    if file_path:
        uploaded_img_cv = cv2.imread(file_path, cv2.IMREAD_COLOR)

        rgb_image = cv2.cvtColor(uploaded_img_cv, cv2.COLOR_BGR2RGB)
        uploaded_img = Image.fromarray(rgb_image)

        uploaded_img.thumbnail((300, 300))  # Resize for display purposes
        uploaded_img_tk = ImageTk.PhotoImage(uploaded_img)
        before_img_label.config(image=uploaded_img_tk)
        before_img_label.image = uploaded_img_tk

        # Save for processing
        global input_image
        input_image = uploaded_img_cv


def process_image():
    """Call processing functions on the uploaded image."""
    global input_image
    if input_image is not None:
        if (operations[0].get()):  # noise removal
            input_image = apply_noise_removal(input_image, (5, 5))

        if (operations[1].get()):  # brightness enhancement
            input_image = adjust_brightness_to_range(input_image)

        if (operations[2].get()):  # contrast enhancement
            input_image = enhance_contrast(input_image)

        if (operations[3].get()):  # chromatic aberration correction
            input_image = CACorrection(input_image)

        if (operations[4].get()):  # scratch removal
            input_image = scratch_removal(input_image)

        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        processed_img = Image.fromarray(rgb_image)
        processed_img.thumbnail((300, 300))
        processed_img_tk = ImageTk.PhotoImage(processed_img)
        after_img_label.config(image=processed_img_tk)
        after_img_label.image = processed_img_tk
        
        processed_img.save("output_image.png")


def continue_processing():
    if (after_img_label.image is not None):
        before_img_label.config(image=after_img_label.image)
        before_img_label.image = after_img_label.image
    after_img_label.config(image=None)
    after_img_label.image = None


# Create the main window
root = tk.Tk()
root.title("FlawFix: Image Processing App")
root.geometry("1000x700")
root.configure(bg="#2c3e50")

# Header with title
header_frame = tk.Frame(root, bg="#34495e", pady=20)
header_frame.pack(fill="both")

header_label = tk.Label(header_frame, text="FlawFix", font=(
    "Helvetica", 20, "bold"), fg="white", bg="#34495e")
header_label.pack()


# Frame for the checkboxes
checkbox_frame = tk.Frame(root, bg="#f0f0f0")
checkbox_frame.pack(pady=20)

# List of options
operationNames = ["noise removal", "brightness enhancement",
                  "color correction", "chromatic aberration correction", "scratch removal"]
operations = []

# Create checkboxes
checkboxes = []
for option in operationNames:
    var = tk.BooleanVar()  # Variable to track the state of the checkbox
    operations.append(var)
    chk = ttk.Checkbutton(checkbox_frame, text=option, variable=var)
    # Align to the left and add some spacing
    chk.pack(side="left", padx=7, pady=2)
    checkboxes.append((var, option))


# Upload Frame
upload_frame = tk.Frame(root, bg="#2c3e50", pady=10)
upload_frame.pack()

instructions_label = tk.Label(upload_frame, text="Upload an image to process:", font=(
    "Arial", 14), fg="white", bg="#2c3e50")
instructions_label.pack(pady=2)

btn_upload = ttk.Button(
    upload_frame, text="Upload Image", command=upload_image)
btn_upload.pack(pady=10)

# Create a frame to hold the two images side by side
images_frame = tk.Frame(root, bg="#2c3e50")
images_frame.pack(pady=20)

before_img_label = tk.Label(images_frame, bg="#2c3e50")
before_img_label.pack(side="left", padx=10)

after_img_label = tk.Label(images_frame, bg="#2c3e50")
after_img_label.pack(side="left", padx=10)

# Process Button
btn_process = ttk.Button(root, text="Process Image", command=process_image)
btn_process.pack(pady=20)

btn_continue = ttk.Button(
    root, text="Continue Processing", command=continue_processing)
btn_continue.pack(pady=10)  # Show the button

# Run the application
root.mainloop()
