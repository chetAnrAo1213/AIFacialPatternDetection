import os
from flask import Flask, request, render_template, send_from_directory
import google.generativeai as genai
from pathlib import Path

# Configure Flask app
app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/'  # Change to use /tmp/ directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists (this is typically not necessary for /tmp/)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Gemini API
genai.configure(api_key="AIzaSyCEfytqyG9hCiKfQae0N_yxh14gRv5gUeg")

def image_format(image_path):
    """Converts image to required format for Gemini API."""
    img = Path(image_path)

    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    if img.suffix in ['.jpeg', '.jpg']:
        mime_type = "image/jpeg"
    elif img.suffix == '.png':
        mime_type = "image/png"
    else:
        raise ValueError("Unsupported image format. Only JPEG and PNG are allowed.")

    return [{"mime_type": mime_type, "data": img.read_bytes()}]

def gemini_output(image_path_1, image_path_2, system_prompt):
    """Handles the communication with Gemini API for facial pattern matching."""
    image_info_1 = image_format(image_path_1)
    image_info_2 = image_format(image_path_2)

    input_prompt = [system_prompt, image_info_1[0], image_info_2[0]]

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(input_prompt)

    return response.text.strip()

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

system_instruction = (
    "You are an expert in analyzing the facial dynamics of two images. "
    "Provide a similarity score between the provided images (range: 0.00 to 100.00). "
    "Respond only with:'Facial Patterns Matched :- (range)'."  # Ensure strict response formatting
)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')

        if file1 and file2:
            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
            file1.save(file1_path)
            file2.save(file2_path)

            # Get the similarity score
            output = gemini_output(file1_path, file2_path, system_prompt=system_instruction)

            return render_template('result.html', file1_filename=file1.filename,
                                   file2_filename=file2.filename, result=output)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
