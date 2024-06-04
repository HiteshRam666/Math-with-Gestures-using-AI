import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
 
# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Math with Gestures", page_icon="🤖")

# Title and description
st.title('Math with Gestures using AI 🤖')
st.markdown("""
    Welcome to the Math with Gestures application! This tool uses AI to recognize hand gestures and solve math problems drawn in the air.
    Simply show a gesture to start drawing and another to get your math problem solved.
""")
 
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("### Live Hand Gesture Recognition")
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
 
with col2:
    st.markdown("### AI Math Solution")
    st.info("Raise four fingers to submit the problem to AI")
    output_text_area = st.empty()
 
genai.configure(api_key="AIzaSyBL-PAo9RzWnU76DkINPjOGQuem14m8jkA")
model = genai.GenerativeModel('gemini-1.5-flash')
 
# Initialize the webcam to capture video
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
 
# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)
 
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None
 
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None: prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (0, 255, 0), 10)
    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(img)
    return current_pos, canvas
 
def sendToAI(model, canvas, fingers):
    if fingers == [0, 1, 1, 1, 1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem, and give proper solution", pil_image])
        return response.text
 
prev_pos = None
canvas = None
image_combined = None
output_text = ""
# Continuously get frames from the webcam
while True:
    success, img = cap.read()
    if canvas is None:
        canvas = np.zeros_like(img)
 
    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)
 
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")
 
    if output_text:
        output_text_area.markdown(f"<p style='text-align: justify;'>{output_text}</p>", unsafe_allow_html=True)
 
    cv2.waitKey(1)
