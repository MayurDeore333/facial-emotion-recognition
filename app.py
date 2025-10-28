from deepface import DeepFace
import gradio as gr

def analyze_emotion(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return f"Predicted Emotion: {emotion}"
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=analyze_emotion,
    inputs=gr.Image(type="numpy", label="Upload Face Image"),
    outputs="text",
    title="Facial Emotion Recognition",
    description="Upload an image to detect the dominant emotion using DeepFace."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=10000)
