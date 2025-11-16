import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.title("🧪 YOLOv8 Microplastic Detection System")

# Load YOLO model
model = YOLO("best.pt")

# MANUAL CLASS LIST (⛔ DO NOT CHANGE ORDER HERE)
CLASSES = ['PET', 'PVC', 'PP', 'HDPE', 'Nylon', 'ABS']

# Harmful Effects
HARMFUL_EFFECTS = {
    "PET": "Releases antimony & BPA – linked to hormonal disruption & cancer.",
    "PVC": "Contains chlorine & phthalates – causes liver damage & cancer risk.",
    "PP": "Low toxicity but creates microfibers entering the food chain.",
    "HDPE": "Leaches endocrine disruptors affecting fertility & growth.",
    "Nylon": "Releases toxic microfibers that harm aquatic organisms.",
    "ABS": "Contains styrene – affects nervous & respiratory systems."
}

# Input Selector
option = st.radio("Select Input Mode:", ["📤 Upload Image", "📷 Use Camera"])


def safe_class_name(cls_index):
    """Safely map YOLO index to correct class name"""
    name_from_list = CLASSES[cls_index] if cls_index < len(CLASSES) else None
    name_from_model = model.names.get(cls_index, None)

    # If model.names version exists & matches a known class, use it
    if name_from_model in CLASSES:
        return name_from_model

    return name_from_list


def process_detection(image):
    results = model(image)
    result_img = results[0].plot()   # annotated output
    st.image(result_img, caption="Detection Result")

    boxes = results[0].boxes
    total = len(boxes)

    if total == 0:
        st.warning("⚠ No microplastic detected")
        return

    class_counts = {}

    for box in boxes:
        cls_index = int(box.cls[0])
        name = safe_class_name(cls_index)

        class_counts[name] = class_counts.get(name, 0) + 1

    st.subheader("📊 Detection Summary")

    percentages = {}

    for cls, count in class_counts.items():
        percent = (count / total) * 100
        percentages[cls] = percent

        st.markdown(f"""
        ### 🧬 {cls}
        🔢 Count: **{count}**  
        📈 Percentage: **{percent:.2f}%**  
        ☣ Harmful Effect: _{HARMFUL_EFFECTS.get(cls, "No known harmful effects.")}_
        """)

    # -------- BAR CHART --------
    st.subheader("📉 Microplastic Percentage Bar Chart")

    fig, ax = plt.subplots()
    ax.bar(percentages.keys(), percentages.values())
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Microplastic Type")
    ax.set_title("Microplastic Composition in Image")
    st.pyplot(fig)


# UPLOAD MODE
if option == "📤 Upload Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image")
        process_detection(image)

# CAMERA MODE
else:
    cam_img = st.camera_input("Take a Picture")
    if cam_img:
        image = Image.open(cam_img)
        process_detection(image)
