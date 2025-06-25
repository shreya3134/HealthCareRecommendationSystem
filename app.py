from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

DISASTER_CLASSES = ['flood', 'fire', 'landslide']
clf_session = ort.InferenceSession("disaster_classifier.onnx")
clf_input_name = clf_session.get_inputs()[0].name

SEG_CLASSES = ["background", "flood", "building", "road", "vegetation"]
seg_session = ort.InferenceSession("floodnet_unet.onnx")
seg_input_name = seg_session.get_inputs()[0].name

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    result_img = None
    error = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            error = "Please upload an image."
        else:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Preprocess
            img = Image.open(filepath).convert('RGB').resize((256, 256))
            img_np = np.array(img).astype(np.float32) / 255.0
            img_input = np.expand_dims(img_np, axis=0)

            # Classification
            clf_output = clf_session.run(None, {clf_input_name: img_input})
            pred_class = np.argmax(clf_output[0])
            raw_pred = DISASTER_CLASSES[pred_class]
            prediction = raw_pred if raw_pred == 'landslide' else 'flood'

            # Segmentation (not shown, but still performed)
            seg_output = seg_session.run(None, {seg_input_name: img_input})[0]
            pred_mask = np.argmax(seg_output[0], axis=-1)

            # Save actual segmentation (not shown)
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(img_np)
            axs[0].set_title("Input Image")
            axs[0].axis("off")
            axs[1].imshow(pred_mask, cmap="tab20", vmin=0, vmax=len(SEG_CLASSES)-1)
            axs[1].set_title("Segmentation")
            axs[1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_FOLDER, "screenshot.png"))
            plt.close()

            # Show default image
            result_img = 'results/default.png'

    return render_template("index.html", prediction=prediction, filename=filename, result_img=result_img, error=error)

if __name__ == '__main__':
    app.run(debug=True)
