import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from lib import *

from PIL import Image
from mushroom_classifier import MushroomClassifier
import wikipedia

st.set_page_config(layout="wide")


@st.experimental_singleton
def init_model():
    return MushroomClassifier(__name__)


def main():
    st.markdown("<h1 style='text-align: center; color: Black; font-size:50pt;'>What the Funghi?</h1>", unsafe_allow_html=True)
    st.write('')
    st.write('')

    # st.text("Choose an image of a mushroom!")
    st.markdown("<p style='color: Black; font-size:14pt;'>Choose an image of a mushroom!</p>",
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['.png', '.jpg'])

    if (uploaded_file is not None):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write('')

        with col2:
            st.image(uploaded_file, caption='Uploaded image', width=600)
            transformed_image = preprocess_input(uploaded_file)

        with col3:
            st.write('')

    checkExpAI = st.checkbox("Select for Explainable AI analysis")
    if st.button("Classify"):
        if uploaded_file is None:
            st.markdown("<h2 style='text-align: center; color: black;'>Please upload an image first!</h2>",
                        unsafe_allow_html=True)
        else:

            certainty, pred_label_idx, y_pred, = classify(transformed_image)
            # cropped_img, grad_fig, occ_fig, wikiresult
            label = CLASSES[pred_label_idx]

            st.markdown("<h2 style='text-align: center; color: black;'>Classification result:</h2>",
                        unsafe_allow_html=True)
            st.write("")

            st.markdown("<h2 style='text-align: center; color: black;'>" + label + "</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: black;'>with a certainty of</p>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: black;'>" + str(certainty) + "</h2>",
                        unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(12, 2))
            ax.bar(np.array(CLASSES), y_pred)
            st.pyplot(fig)

            wiki_container = st.container()

            with wiki_container:
                st.write("")
                st.write("")
                st.write("")
                st.markdown("<h2 style='text-align: center; color: black;'>Wikipedia entry</h2>",
                            unsafe_allow_html=True)
                st.write("")
                st.write(wikipedia.summary(label, sentences=5, auto_suggest=False))

            if checkExpAI:
                st.write("")
                st.write("")
                st.write("")
                st.markdown("<h2 style='text-align: center; color: black;'>Explainable AI:</h2>",
                            unsafe_allow_html=True)
                st.write("")
                st.write("")
                st.write("")

                grad_fig, occ_fig = exp_AI(transformed_image, pred_label_idx)
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("      Model input image")
                    cropped_img = model.crop_transformations(Image.open(uploaded_file))
                    st.image(cropped_img, use_column_width=True)
                with col2:
                    st.subheader("      Gradient-based attribution")
                    st.pyplot(grad_fig)
                with col3:
                    st.subheader("      Occlusion-based attribution")
                    st.pyplot(occ_fig)


def classify(transformed_img):
    model_input = transformed_img.unsqueeze(0)
    model_output = model.shroom_model(model_input)
    model_output = torch.nn.functional.softmax(model_output, dim=1)
    prediction_score, pred_label_idx = torch.topk(model_output, 1)
    pred_label_idx.squeeze_()

    certainty = prediction_score.squeeze().item()

    y_pred = model_output.detach().numpy().flatten()
    return certainty, pred_label_idx.squeeze_(), y_pred,


def exp_AI(transformed_img, pred_label_idx):
    model_input = transformed_img.unsqueeze(0)
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)

    torch.manual_seed(0)
    np.random.seed(0)

    gradient_shap = GradientShap(model.shroom_model)
    rand_img_dist = torch.cat([model_input * 0, model_input * 1])
    attributions_gs = gradient_shap.attribute(model_input,
                                              n_samples=50,
                                              stdevs=0.0001,
                                              baselines=rand_img_dist,
                                              target=pred_label_idx)
    grad_fig = viz.visualize_image_attr_multiple(
        np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["heat_map"],
        ["absolute_value"],
        cmap=default_cmap,
        show_colorbar=True,
        use_pyplot=False)

    occlusion = Occlusion(model.shroom_model)

    attributions_occ = occlusion.attribute(model_input,
                                           strides=(3, 8, 8),
                                           target=pred_label_idx,
                                           sliding_window_shapes=(3, 15, 15),
                                           baselines=0)

    occ_fig = viz.visualize_image_attr_multiple(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["heat_map"],
        ["positive"],
        show_colorbar=True,
        outlier_perc=2,
        use_pyplot=False)

    # st.pyplot(grad_fig[0])
    # st.pyplot(occ_fig[0])

    #
    # input_img_path = os.path.join('static', 'results', f'{wz_image.filename[:-4]}_inp.jpg')
    # if os.path.isfile(input_img_path):
    #     os.remove(input_img_path)
    #
    # bbox = occ_fig[0].get_window_extent().transformed(occ_fig[0].dpi_scale_trans.inverted())
    # width, height = bbox.width * occ_fig[0].dpi, bbox.height * occ_fig[0].dpi
    # # print("hi",width, height)
    # cropped_img.resize((int(cropped_img.width * 1.5), int(cropped_img.height * 1.5))).save(input_img_path)
    #
    # occ_output_img_path = os.path.join('static', 'results', f'{wz_image.filename[:-4]}_occ.jpg')
    # if os.path.isfile(occ_output_img_path):
    #     os.remove(occ_output_img_path)
    # occ_fig[0].savefig(occ_output_img_path, bbox_inches='tight')
    #
    # grad_output_img_path = os.path.join('static', 'results', f'{wz_image.filename[:-4]}_grad.jpg')
    # if os.path.isfile(grad_output_img_path):
    #     os.remove(grad_output_img_path)
    # grad_fig[0].savefig(grad_output_img_path, bbox_inches='tight')

    # wikiresult = wikipedia.summary(label, sentences=5, auto_suggest=False)

    return grad_fig[0], occ_fig[0]
    # output = {
    #     'image1': input_img_path,
    #     'image2': occ_output_img_path,
    #     'image3': grad_output_img_path,
    #     'name': label,
    #     'text': f'Certainty: {certainty}',
    #     'wiki': wikiresult
    # }
    #
    # return output


def preprocess_input(image):
    image = Image.open(image)
    cropped_img = model.crop_transformations(image)
    transformed_img = model.norm_transformations(cropped_img)
    return transformed_img


if __name__ == "__main__":
    model = init_model()
    main()
