import streamlit as st
import base64
import os

#User accession page
#Setting background image
def set_background(image_file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_file_name)
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("national-cancer-institute-I-F4ibaAtLA-unsplash_copy2.jpg")

#Title
st.markdown("""
    <h1 style='text-align: center; ; font-family: Helvetica; margin-bottom: 15px'>Breast Tumor classifier</h1>
""", unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center; font-family: Helvetica'>A deep learning-based AI model that analyzes microscopic images of breast tumor tissue to distinguish between benign and malignant samples.</p>
""", unsafe_allow_html=True)

#Central button (trying)
#Injecting custom CSS to style the button
st.markdown("""
    <style>
    div.stButton > button {
        font-size: 20px;
        padding: 18px 18px;
        background-color: #750000;
        color: white;
        border-radius: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Center the button using columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3:
    center_button = st.button("Start analysis")

#Example of images with headers
st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.image("C:/Users/Eleonora/OneDrive/Desktop/AI LAB PROJECT/SOB_B_A-14-22549CD-40-008.png", caption="Benign Sample (40X)", use_container_width=True)

with col2:
    st.image("C:/Users/Eleonora/OneDrive/Desktop/AI LAB PROJECT/SOB_M_MC-14-18842D-40-011.png", caption="Malignant Sample (40X)", use_container_width=True)

st.markdown("<br>", unsafe_allow_html= True)

with st.expander("How did we train the model?"):
    st.markdown("bla bla bla")

with st.expander("What results did we obtain?"):
    st.markdown("bla bla bla")

with st.expander("Why 40X resolution?"):
    st.markdown("In microscopy, **40X** magnification enlarges tissue structures 40 times their original size. " \
    "In this project, we used only the 40X resolution images for model training and evaluation.")

with st.expander("Learn more"):
    st.markdown("Breast cancer is the most prevalent cancer worldwide, killing roughly 600.000 women annually. " \
    "With this grim trend on the rise in the last few years, it is more important now than ever before to identify cancer in time to be able to combat it for as many patients as possible. " \
    "This is why we developed our MILF (Malignant Interference via Learned Features) model as an image classifier that can recognize malignant features of breast cancer based on scan and text input data.")

st.markdown("<p style= 'font-family: Helvetica;'>Dataset specifics (BreakHis: Breast Cancer Histopathological Database)</p>", unsafe_allow_html=True)

st.markdown("""
- **Total images**: 9,109  
- **Benign samples**: 2,480  
- **Malignant samples**: 5,429  
- **Magnifications**: 40X, 100X, 200X, 400X  
""")

#Interactive gallery (fake next and previous buttons, finto carosello?)
#st.image(["img1.png", "img2.png", "img3.png"], caption=["Benign", "Malignant", "Unknown"], width=200)

#If user presses start, going to next phase
if st.session_state.get("page") == "analyze":
    st.switch_page("upload_and_classify.py")  # You must create this file with the logic for upload and model use


