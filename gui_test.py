import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

class ModernYOLOVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Myanmar License Plate Detector (YOLOv8)")
        self.root.geometry("1000x700")
        self.root.configure(bg="#111827")  # Dark Background
        
        # Default Model Path
        self.model_path = "D:/hwa/github project/myanmar-car-license-plate/runs/detect/myanmar-car-plate/yolov8_training/weights/best.pt"
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            self.status_msg = "Model Loaded: ready to detect! 🟢"
            self.status_color = "#10b981"
        else:
            self.model = None
            self.status_msg = "⚠️ best.pt ကို ရှာမတွေ့ပါ။ Model အရင်ရွေးပေးပါ။"
            self.status_color = "#ef4444"

        self.create_widgets()

    def create_widgets(self):
        # --- HEADER SECTION ---
        header_frame = tk.Frame(self.root, bg="#1f2937", height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_lbl = tk.Label(
            header_frame, 
            text="🚗 MYANMAR VEHICLE & LICENSE PLATE DETECTOR", 
            fg="#ffffff", 
            bg="#1f2937", 
            font=("Segoe UI", 14, "bold")
        )
        title_lbl.pack(side=tk.LEFT, padx=25, pady=20)

        # Status Badge
        self.lbl_status = tk.Label(
            header_frame, 
            text=self.status_msg, 
            fg=self.status_color, 
            bg="#111827", 
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=6,
            relief="flat"
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=25, pady=20)

        # --- CONTROL BUTTONS SECTION ---
        btn_frame = tk.Frame(self.root, bg="#111827", pady=15)
        btn_frame.pack(fill=tk.X)

        # Select Model Button (Indigo)
        self.btn_model = tk.Button(
            btn_frame, 
            text="⚙️ Select Custom Model (.pt)", 
            command=self.load_custom_model,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#4f46e5",
            activeforeground="#ffffff",
            activebackground="#4338ca",
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2"
        )
        self.btn_model.pack(side=tk.LEFT, padx=25)
        self.bind_hover_events(self.btn_model, "#4f46e5", "#4338ca")

        # Select Image Button (Emerald Green)
        self.btn_detect = tk.Button(
            btn_frame, 
            text="🖼️ Upload Image & Detect", 
            command=self.detect_image,
            font=("Segoe UI", 10, "bold"),
            fg="#ffffff",
            bg="#10b981",
            activeforeground="#ffffff",
            activebackground="#059669",
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2"
        )
        self.btn_detect.pack(side=tk.LEFT, padx=10)
        self.bind_hover_events(self.btn_detect, "#10b981", "#059669")

        # --- DISPLAY PANEL (IMAGE VIEWER) ---
        display_frame = tk.Frame(self.root, bg="#111827")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=10)

        # Left Panel - Original Image Card
        left_card = tk.Frame(display_frame, bg="#1f2937", bd=0, highlightthickness=1, highlightbackground="#374151")
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_header = tk.Label(left_card, text="Original Image", fg="#9ca3af", bg="#1f2937", font=("Segoe UI", 11, "bold"))
        left_header.pack(anchor="w", padx=15, pady=10)

        self.lbl_orig_img = tk.Label(left_card, text="ပုံရွေးချယ်ရန် စောင့်ဆိုင်းနေပါသည်...", fg="#6b7280", bg="#111827", font=("Segoe UI", 10))
        self.lbl_orig_img.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Right Panel - Detected Image Card
        right_card = tk.Frame(display_frame, bg="#1f2937", bd=0, highlightthickness=1, highlightbackground="#374151")
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_header = tk.Label(right_card, text="Detection Result", fg="#9ca3af", bg="#1f2937", font=("Segoe UI", 11, "bold"))
        right_header.pack(anchor="w", padx=15, pady=10)

        self.lbl_detect_img = tk.Label(right_card, text="Detect လုပ်ထားခြင်း မရှိသေးပါ...", fg="#6b7280", bg="#111827", font=("Segoe UI", 10))
        self.lbl_detect_img.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

    # Hover Animations for Buttons
    def bind_hover_events(self, button, normal_color, hover_color):
        button.bind("<Enter>", lambda e: button.config(bg=hover_color))
        button.bind("<Leave>", lambda e: button.config(bg=normal_color))

    def load_custom_model(self):
        file_selected = filedialog.askopenfilename(
            title="Select YOLO Model Weights",
            filetypes=[("PyTorch Weights", "*.pt")]
        )
        if file_selected:
            self.model = YOLO(file_selected)
            self.lbl_status.config(
                text=f"Loaded: {os.path.basename(file_selected)} ✅", 
                fg="#10b981"
            )

    def detect_image(self):
        if self.model is None:
            messagebox.showerror("Error", "YOLO Model (best.pt) ကို အရင်ရွေးချယ်ပေးပါ!")
            return

        img_path = filedialog.askopenfilename(
            title="Select Image to Detect",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if not img_path:
            return

        # ၁။ Original Image ပြသခြင်း
        self.show_image_on_label(img_path, self.lbl_orig_img)

        # ၂။ RTX 5070 Laptop GPU ဖြင့် Predict လုပ်ခြင်း
        results = self.model.predict(source=img_path, conf=0.25, device=0)
        
        # ၃။ Results ပုံကို Plot လုပ်ပြီး UI ပေါ်တင်ခြင်း
        res_plotted = results[0].plot() 
        detected_img = Image.fromarray(res_plotted[..., ::-1]) 
        self.show_image_on_label(detected_img, self.lbl_detect_img, is_pil=True)
        
        # Status update
        num_boxes = len(results[0].boxes)
        if num_boxes > 0:
            self.lbl_status.config(text=f"ရှာဖွေတွေ့ရှိမှု - {num_boxes} ခု 🎯", fg="#10b981")
        else:
            self.lbl_status.config(text="ဘာမှ ရှာမတွေ့ပါ ❌", fg="#ef4444")

    def show_image_on_label(self, img_source, label, is_pil=False):
        if is_pil:
            img = img_source
        else:
            img = Image.open(img_source)

        # UI Canvas Size အလိုက် အချိုးကျ Resize လုပ်ခြင်း
        canvas_width = 440
        canvas_height = 420
        
        img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo, text="", bg="#111827")
        label.image = photo  # Keep reference

if __name__ == '__main__':
    root = tk.Tk()
    app = ModernYOLOVisualizer(root)
    root.mainloop()