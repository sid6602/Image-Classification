import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load the pre-trained deep learning model
model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=True)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    return img

def main():
    st.title('Image Classification')

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Save the uploaded image
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.read())
        
        # Preprocess the image
        img = preprocess_image("uploaded_image.jpg")

        # Use the model to make predictions
        predictions = model.predict(np.expand_dims(img, axis=0))
        decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(predictions, top=5)[0]

        # Display the top predicted labels
        st.image("uploaded_image.jpg", caption="Uploaded Image", use_column_width=True)
        st.subheader("Top Predicted Labels:")
        for i, (_, label, score) in enumerate(decoded_predictions):
            st.write(f"{i + 1}. {label} ({score:.2f})")

if __name__ == '__main__':
    main()
