from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import pickle

app = Flask(__name__)

# Load the trained Keras model
model_keras = tf.keras.models.load_model("model.keras")

# Load the pickle model
model_pickle = pickle.load(open("model1.pkl", "rb"))

output_class = ["batteries", "bottles", "clothes", "e-waste", "glass", "light bulbs", "metal", "organic", "paper", "plastic"]

recycling_suggestions = {
    "batteries": "Collect used batteries, store them in a dry place, and take them to a local recycling center or retailer for proper disposal.",
    "bottles": "Rinse bottles, remove caps. Check local guidelines. Place in recycling bin. Help reduce waste and conserve resources.",
    "clothes": "Donate old clothes to thrift stores, shelters, or textile recycling centers to reduce waste and help those in need.",
    "e-waste": "To recycle e-waste, locate a certified collection center or electronics retailer that accepts discarded devices for responsible disposal.",
    "glass": "Collect clean glass containers, separate by color, remove lids/caps, rinse, and place in recycling bin.",
    "light bulbs": "Take used lightbulbs to a recycling center or participating store. Check local guidelines for safe disposal of CFLs and LEDs.",
    "metal": "Collect metal items, like cans and wires. Take them to a recycling center or curbside pickup for processing.",
    "organic": "Collect food scraps, yard waste. Compost at home or use municipal composting. Nutrients return to soil, reduce landfill waste.",
    "paper": " Collect paper, remove contaminants, sort by type, shred or pulp, mix with water, form new sheets, dry, and reuse.",
    "plastic": "Collect plastic waste, clean and sort by type. Check local recycling guidelines. Drop off at recycling center. Avoid contamination."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/college_predict', methods=['POST'])
def college_predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model_pickle.predict(features)
    
    return render_template("index.html",prediction_text="The college suggested is: {}".format(prediction))
   

@app.route('/waste_predict', methods=['POST'])
def waste_predict():
    try:
        # Get the uploaded image
        img = request.files['image']
        img_path = "static/uploaded_image.jpg"  # Save the uploaded image
        img.save(img_path)

        test_image = image.load_img(img_path, target_size=(224, 224))
        test_image = image.img_to_array(test_image) / 255
        test_image = np.expand_dims(test_image, axis=0)

        predicted_array = model_keras.predict(test_image)
        predicted_value = output_class[np.argmax(predicted_array)]
        predicted_accuracy = round(np.max(predicted_array) * 100, 2)

        recycling_suggestion = recycling_suggestions.get(predicted_value, "Sorry, no recycling suggestion available for this item.")

        return render_template('index.html', prediction_value=predicted_value,
                               prediction_accuracy=predicted_accuracy, recycling_suggestion=recycling_suggestion,
                               image_path=img_path)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)