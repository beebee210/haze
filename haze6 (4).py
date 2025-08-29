import numpy as np
from PIL import Image
import streamlit as st
import cv2
import time

# ===================== Account System =====================
def account_system():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "users" not in st.session_state:
        st.session_state.users = {}  # {username: password}

    if not st.session_state.authenticated:
        st.title("🔐 Login to ClearSky.AI")

        choice = st.radio("Choose an option", ["Login", "Create Account"])

        if choice == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if username in st.session_state.users and st.session_state.users[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.current_user = username
                    st.success("✅ Login successful!")
                else:
                    st.error("❌ Invalid username or password")
            st.stop()

        elif choice == "Create Account":
            new_user = st.text_input("Choose a username")
            new_pass = st.text_input("Choose a password", type="password")
            confirm_pass = st.text_input("Confirm password", type="password")

            st.markdown("### 📜 PDPA Terms & Conditions")
            st.write("""
            By creating an account, you agree that:
            - This demo app does **not store or share** your personal data permanently.  
            - Uploaded images are processed only for haze detection and are **not saved**.  
            - Any image containing **faces** will be blurred for privacy.  
            - You can stop using this service at any time.  
            """)

            agree = st.checkbox("✅ I have read and agree to the above terms.")

            if st.button("Create Account"):
                if not new_user or not new_pass:
                    st.error("❌ Username and password cannot be empty")
                elif new_pass != confirm_pass:
                    st.error("❌ Passwords do not match")
                elif new_user in st.session_state.users:
                    st.error("❌ Username already exists")
                elif not agree:
                    st.error("❌ You must agree to the PDPA terms to create an account")
                else:
                    st.session_state.users[new_user] = new_pass
                    st.success("✅ Account created! Please login now.")
            st.stop()

# ===================== Haze Detection =====================
def get_dark_channel(I, w=15):
    min_channel = np.min(I, axis=2)
    kernel = np.ones((w, w), np.uint8)
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def analyze_dark_channel(dark_channel):
    mean_val = np.mean(dark_channel)
    if mean_val < 80:
        return "No Haze", mean_val
    elif 80 <= mean_val < 120:
        return "Light Haze", mean_val
    else:
        return "Heavy Haze", mean_val

# ===================== Face Masking =====================
def mask_faces(image_np):
    """Detects faces and blurs them for privacy."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face_region = image_np[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)
        image_np[y:y+h, x:x+w] = blurred_face

    return image_np, len(faces)

# ===================== Webcam Function =====================
def realtime_camera(username):
    frame_slot = st.empty()
    haze_label = st.empty()
    haze_value = st.empty()
    warning_box = st.empty()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        st.error("❌ Could not access webcam.")
        return

    haze_values = []
    run = st.checkbox("▶️ Start Camera")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ No frame captured.")
            time.sleep(0.2)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ Blur faces
        frame_rgb, face_count = mask_faces(frame_rgb)
        if face_count > 0:
            warning_box.warning(f"⚠️ {face_count} face(s) detected and blurred for privacy.")

        # ✅ Haze detection
        I_norm = cv2.normalize(frame_rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        I = np.asarray(I_norm, dtype=np.float64)

        dark = get_dark_channel(I, w=15)
        _, mean_val = analyze_dark_channel(dark)
        haze_values.append(mean_val)

        if len(haze_values) >= 5:
            avg_val = np.mean(haze_values)
            haze_result, _ = analyze_dark_channel(np.full_like(dark, avg_val))
            haze_values.clear()

            frame_slot.image(frame_rgb, caption="Live Feed (320x240)", channels="RGB")
            haze_label.markdown(f"### 🧪 Haze Detection: **{haze_result}**")
            haze_value.write(f"📊 Avg. dark channel intensity: {avg_val:.2f}")

        time.sleep(0.2)

    cap.release()

# ===================== Accuracy (No sklearn) =====================
def accuracy_score_manual(true_labels, predictions):
    correct = sum(t == p for t, p in zip(true_labels, predictions))
    return correct / len(true_labels) if true_labels else 0

# ===================== Label Normalization =====================
def normalize_label(label):
    """Map any haze string to a standard category"""
    if "No" in label:
        return "No Haze"
    elif "Light" in label:
        return "Light Haze"
    elif "Heavy" in label:
        return "Heavy Haze"
    else:
        return label.strip()

# ===================== Evaluation Function =====================
def evaluate_model(test_images, test_labels):
    predictions = []
    for img, true_label in zip(test_images, test_labels):
        img_np = np.array(img, dtype=np.float64)
        img_norm = cv2.normalize(img_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dark = get_dark_channel(img_norm, w=15)
        haze_result, _ = analyze_dark_channel(dark)

        predictions.append(normalize_label(haze_result))

    # ✅ Normalize true labels too
    normalized_true = [normalize_label(lbl) for lbl in test_labels]

    acc = accuracy_score_manual(normalized_true, predictions) * 100
    return acc, predictions, normalized_true

# ===================== Streamlit UI =====================
st.set_page_config(page_title="🌤️ Haze Detection AI", layout="centered")

# Require login first
account_system()

st.title("🌤️ Haze Detection AI")
st.markdown("Welcome to **ClearSky.AI** — Detect haze in real time or from images.")

# Sidebar menu
option = st.sidebar.radio("Choose Mode", ("📁 Upload Image", "📹 Real-Time Camera", "📊 Evaluate Model"))

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        # ✅ Blur faces
        img_np, face_count = mask_faces(img_np)
        if face_count > 0:
            st.warning(f"⚠️ {face_count} face(s) detected and blurred for privacy.")

        with st.spinner("🔍 Analyzing image for haze..."):
            img_np_float = np.array(img_np, dtype=np.float64)
            img_norm = cv2.normalize(img_np_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dark = get_dark_channel(img_norm, w=15)
            haze_result, mean_val = analyze_dark_channel(dark)

            st.image(img_np, caption="Uploaded Image (faces blurred)")
            st.markdown(f"### 🧪 Haze Detection: **{haze_result}**")
            st.write(f"📊 Avg. dark channel intensity: {mean_val:.2f}")

elif option == "📹 Real-Time Camera":
    st.markdown("### 📸 Real-Time Webcam Feed")
    realtime_camera(username=st.session_state.get("current_user"))

elif option == "📊 Evaluate Model":
    st.markdown("### 📊 Model Evaluation")
    uploaded_files = st.file_uploader("Upload labeled test images", type=["jpg", "png"], accept_multiple_files=True)
    labels_input = st.text_area("Enter labels for each image (comma separated: No Haze, Light Haze, Heavy Haze)")

    if uploaded_files and labels_input:
        test_images = [Image.open(f).convert("RGB") for f in uploaded_files]
        test_labels = [label.strip() for label in labels_input.split(",")]

        if len(test_images) != len(test_labels):
            st.error("❌ Number of images and labels must match.")
        else:
            acc, preds, normalized_true = evaluate_model(test_images, test_labels)
            st.success(f"✅ Model Accuracy: {acc:.2f}%")

            # Show results
            for i, (img, pred, true) in enumerate(zip(test_images, preds, normalized_true)):
                st.image(img, caption=f"Image {i+1}: Predicted = {pred}, True = {true}")
