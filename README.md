# Math with Gestures using AI 🤖

Welcome to the **Math with Gestures** application! This tool uses AI to recognize hand gestures and solve math problems drawn in the air. Simply show a gesture to start drawing and another to get your math problem solved. 

## 🎉 Features

- **Live Hand Gesture Recognition**: Draw mathematical problems in the air using hand gestures.
- **AI Math Solution**: Solve the problems instantly using AI-powered models.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Make sure you have the following installed:

- Python 3.7+
- OpenCV
- Streamlit
- NumPy
- Pillow
- cvzone
- google-generativeai

You can install the necessary Python packages using pip:

```sh
pip install opencv-python-headless streamlit numpy pillow cvzone google-generativeai
```

### Installation

1. **Clone the repository**:

    ```sh
    git clone https://github.com/your-username/math-with-gestures.git
    cd math-with-gestures
    ```

2. **Run the application**:

    ```sh
    streamlit run app.py
    ```

## 🧠 How It Works

1. **Hand Gesture Recognition**: The application uses `cvzone`'s HandDetector to recognize hand gestures.
2. **Drawing and Clearing Canvas**: 
   - **Index Finger Up**: Draw on the canvas.
   - **All Fingers Up**: Clear the canvas.
3. **Submitting to AI**: Raise four fingers to submit the problem to the AI for a solution.

## 🤝 Contributing

Contributions are welcome! Feel free to open a pull request or issue.


## 📬 Contact

For any questions, feel free to reach out to [hiteshram321@gmail.com](mailto:hiteshram321@gmail.com).

---

**Enjoy solving math problems with gestures!** ✨🖐️

---
