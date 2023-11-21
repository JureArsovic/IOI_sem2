import os
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image

app = Flask(__name__)

# Define the directory where uploaded images will be stored temporarily
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the UPLOAD_FOLDER directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load a pre-trained PyTorch model for feature extraction
model = torch.hub.load(
    'pytorch/vision',
    'mobilenet_v2',
    pretrained=True
)
model.eval()  # Set the model to evaluation mode

# Define image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

#TEST COMMIT


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image to the UPLOAD_FOLDER
        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            file.filename
        )
        file.save(file_path)

        # Open and preprocess the user's uploaded image
        user_image = Image.open(file_path)
        user_image = preprocess(user_image)
        user_image = user_image.unsqueeze(0)  # Add a batch dimension

        # TODO: Implement feature extraction, similarity calculation,
        # and result selection using PyTorch here

        # After processing, render the result template with appropriate data
        return render_template(
            'result.html',
            result_image='path_to_result_image',
            result_title='Movie Title',
            result_score='Similarity Score'
        )


if __name__ == '__main__':
    app.run(debug=True)
