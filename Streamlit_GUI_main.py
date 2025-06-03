import streamlit as st
import base64
import os
from streamlit.components.v1 import html

#SETTING BACKROUND IMAGE
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

set_background("abstract-digital-grid-black-background.jpg")



#TITLE 
st.markdown("""
    <h1 style='text-align: center; ; font-family: Helvetica; margin-bottom: 15px'>Breast Tumor classifier</h1>
""", unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center; font-family: Helvetica'>A deep learning-based AI model that analyzes microscopic images of breast tumor tissue to distinguish between benign and malignant samples.</p>
""", unsafe_allow_html=True)



#CENTRAL BUTTON
#Injecting custom CSS to style the button
st.markdown("""
    <style>
    div.stButton > button {
        font-size: 20px;
        padding: 18px 18px;
        background-color: #025f5f;
        color: white;
        border-radius: 15px;
    }
    </style>
""", unsafe_allow_html=True)

#Centering the button using columns
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


#JS to redirect to the other page after clicking the button
#Avoiding st.switch_page()
def nav_page(page_name, timeout_secs=3):
    nav_script = f"""
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {{
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {{
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {{
                        links[i].click();
                        return;
                    }}
                }}
                var elapsed = new Date() - start_time;
                if (elapsed < timeout_secs * 1000) {{
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                }} else {{
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }}
            }}
            window.addEventListener("load", function() {{
                attempt_nav_page("{page_name}", new Date(), {timeout_secs});
            }});
        </script>
    """
    html(nav_script)


if center_button:
    nav_page("Upload_and_classifyGUI")


#IMAGES WITH HEADERS
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.image("SOB_B_A-14-22549CD-40-008.png", caption="Benign Sample (40X)", use_container_width=True)

with col2:
    st.image("SOB_M_MC-14-18842D-40-011.png", caption="Malignant Sample (40X)", use_container_width=True)

st.markdown("<br>", unsafe_allow_html= True)

#INFO AND QUESTIONS
with st.expander("Why focusing on breast cancer?"):
    st.markdown("Breast cancer is the most prevalent cancer worldwide, killing roughly 600.000 women annually." \
    "With this grim trend on the rise in the last few years, it is more important now than ever before to identify cancer in time to be able to combat it for as many patients as possible." \
    "This is why we developed our MILF (Malignant Interference via Learned Features) model as an image classifier that can recognize malignant features of breast cancer based on scan and text input data.")

with st.expander("What data powers our model?"):
    st.markdown("We used the breast cancer image classification dataset (BreakHis) compiled by Spanhol et al. " \
    "The data is in the form of images from slides of breast tissue, classified into benign and malignant samples. " \
    "There are 4 subtypes of benign tumors (namely adenosis, fibroadenoma, phyllodes tumor and tubular adenoma) and the same amount of malignant subtypes (ductal carcinoma, lobular carcinoma, mucinous carcinoma and papillary carcinoma)." \
    "In total, it contains 7909 images. Each subtype is captured under 40X, 100X, 200X and 400X magnification and comes from 82 patients. " \
    "There are approximately 31% (2480) of benign and 69% (5429) of malignant samples.")

with st.expander("How did we develop the AI system?"):
    st.markdown("In this project we developed a breast cancer histopathological image classification pipeline using a ResNet-50-based deep learning model, trained across multiple magnification levels. " \
    "By using transfer learning, data augmentation techniques and patient-wise splitting, we ensured both performance and generalizability." \
    "Among various configurations, we found that the 100X magnification model with batch size of 64 and 600 images per class achieved the most balanced performance. " \
    "Additionally, the integration of Grad-CAM provided valuable insights into the regions that influenced predictions." \
    "These results show the potential of deep learning to support clinical diagnosis.")

with st.expander("Full project report (PDF)"):
    st.markdown("""
    You can download and read our complete project paper [here](https://github.com/Elelia04/Breast_tumor_classifier/raw/main/MILF_report.pdf).
    """)

st.markdown("<br>", unsafe_allow_html= True)

st.markdown("Dataset specifics (BreakHis):")

st.markdown("""
- **Total images**: 9,109  
- **Benign samples**: 2,480  
- **Malignant samples**: 5,429  
- **Magnifications**: 40X, 100X, 200X, 400X  
""")


#SIDEBAR COLOUR
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.6);  
        backdrop-filter: blur(4px);           
        border-right: 1px solid rgba(255,255,255,0.1);
        color: white;
    }

    [data-testid="stSidebarNav"] > ul {
        padding-top: 30px;
    }

    [data-testid="stSidebarNav"] > ul > li {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        padding: 8px;
    }

    [data-testid="stSidebarNav"] > ul > li:hover {
        background-color: rgba(0,255,170,0.15);
        border-radius: 8px;
        transition: 0.3s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)
