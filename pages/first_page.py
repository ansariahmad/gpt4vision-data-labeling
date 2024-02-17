import streamlit as st
import os
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai.client.model import Model
from clarifai.client.input import Inputs
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont
from os import listdir
from os.path import splitext

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

st.title("Data Labeling using General Object Detection Model and GPT4 Vision")

def main():
    classes_done = []
    with st.sidebar:
        st.subheader('Add your Clarifai PAT.')
        # clarifai_pat = st.text_input('Clarifai PAT:', type='password')
        clarifai_pat = "b0832f0bdcc14c4ba5b57ed5e3401f05"
    if not clarifai_pat:
        st.warning('Please enter your PAT to continue!', icon='⚠️')
        return
    else:
        os.environ['CLARIFAI_PAT'] = clarifai_pat

    # Step 2: Allow user to upload an image
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is None:
        st.warning("Please upload an image!")
        return
    
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    
    image = Image.open(uploaded_file)
    image = image.convert('RGBA')
    with open(uploaded_file.name, "rb") as f:
        file_bytes = f.read()
    

    # Step 3: Allow user to add labels of the image
    st.subheader("Add Labels to the Image")
    labels = st.text_input("Enter label(s) separated by comma (,)")
    labels_list = [label.strip() for label in labels.split(",") if label.strip()]
    
    if not labels_list:
        st.warning("Please add label(s) to proceed!")
        return

    # Step 4: A button to detect objects
    if st.button("Detect Objects"):
        detector_model = Model("https://clarifai.com/clarifai/main/models/objectness-detector")
        
        # prediction_response = detector_model.predict_by_filepath(uploaded_file.read(), input_type="image")
        prediction_response = detector_model.predict_by_bytes(uploaded_file.getvalue(),input_type="image")

        regions = prediction_response.outputs[0].data.regions
        model_url = "https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision"
        classes = ['Ferrari 812', 'Volkswagen Beetle', 'BMW M5', 'Honda Civic']
        threshold = 0.99

        draw = ImageDraw.Draw(image)

        for region in regions:
            top_row = round(region.region_info.bounding_box.top_row, 3)
            left_col = round(region.region_info.bounding_box.left_col, 3)
            bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
            right_col = round(region.region_info.bounding_box.right_col, 3)

            for concept in region.data.concepts:
                prompt = f"Label the object in the Bounding Box region: ({top_row}, {left_col}, {bottom_row}, {right_col}) with one word {labels_list}"

                inference_params = dict(temperature=0.2, max_tokens=100)
                model_prediction = Model("https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision").predict(inputs = [Inputs.get_multimodal_input(input_id="", image_bytes = file_bytes, raw_text=prompt)], inference_params=inference_params)

                # model_prediction = Model(model_url).predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)

                concept_name = model_prediction.outputs[0].data.text.raw
                value = round(concept.value, 4)

                if value > threshold:
                    if concept_name in classes_done:
                        continue
                    classes_done.append(concept_name)
                    top_row = top_row * image.height
                    left_col = left_col * image.width
                    bottom_row = bottom_row * image.height
                    right_col = right_col * image.width

                    draw.rectangle([(int(left_col), int(top_row)), (int(right_col), int(bottom_row))],
                                    outline=(36, 255, 12), width=2)

                    font = ImageFont.load_default()
                    draw.text((int(left_col), int(top_row - 15)), concept_name, font=font, fill=(36, 255, 12))

        st.image(image, caption='Image with Label', channels='BGR', use_column_width=True)
        os.remove(uploaded_file.name)

if __name__ == '__main__':
    main()

