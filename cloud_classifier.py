import streamlit as st
from fastai.vision.all import *
import pandas as pd
import plotly.express as px

learn_inf = load_learner('noaa_model.pkl', cpu=True)

st.title("Cloud Classifier App :cloud:")
multi = '''Not sure what's in the sky? Show us the picture and we'll be happy to tell you what cloud is there!'''
st.markdown(multi)
cloud_desc = {
    'cirrus': 'Detached clouds in the form of white, delicate filaments, mostly in patches or narrow bands. They may have a fibrous (hair-like) and/or silky sheen appearance.',
    'cirrocumulus': "Thin and white, these clouds look like a patchy sheet or layer arranged somewhat-regularly into grains or ripples without shading. Most of these elements have an apparent width of less than one degree (approximately width of the little finger held at arm's length).",
    'cirrostratus': 'Transparent, whitish veil-like clouds with a fibrous (hair-like) or smooth appearance. A sheet of cirrostratus is very extensive and can cover the whole sky.',
    'altocumulus': 'White and/or gray patchy, sheet, or layered clouds generally composed of laminae (plates), rounded masses, or rolls. They may be partly fibrous or diffuse and may or may not be merged.',
    'altostratus': 'Gray or bluish cloud sheets or layers of striated or fibrous clouds that totally or partially cover the sky. They are thin enough to regularly reveal the sun as if seen through ground glass.',
    'nimbostratus': 'Resulting from thickening altostratus, this is a dark gray cloud layer diffused by falling rain or snow. It is thick enough throughout to blot out the sun. Low, ragged clouds frequently occur beneath this cloud and sometimes merge with its base.',
    'cumulus': 'Detached, generally dense clouds and with sharp outlines that develop vertically in the form of rising mounds, domes, or towers with bulging upper parts often resembling a cauliflower.',
    'cumulonimbus': 'The thunderstorm cloud, this is a heavy and dense cloud in the form of a mountain or huge tower. The upper portion is usually smoothed, fibrous, or striated and nearly always flattened in the shape of an anvil or vast plume.',
    'stratocumulus': 'Gray or whitish patchy, sheet, or layered clouds that almost always have dark tessellations (honeycomb appearance), rounded masses, or rolls. Except for virga, they are non-fibrous and may or may not be merged.',
    'stratus': 'A generally gray cloud layer with a uniform base which may, if thick enough, produce drizzle, ice prisms, or snow grains. When the sun is visible through this cloud, its outline is clearly discernible.'    
}

if 'clicked' and 'camera' not in st.session_state:
    st.session_state.clicked = False
    st.session_state.camera = False

def set_clicked():
    st.session_state.clicked = not(st.session_state.clicked)
    
def set_camera():
    st.session_state.camera = not(st.session_state.camera)

col1, col2 = st.columns([1,1])
with col1:
        st.button("Upload Photo", type="primary", use_container_width=True, on_click=set_clicked)
with col2: 
        st.button("Take Photo", type="primary", use_container_width=True, on_click=set_camera)

if st.session_state.clicked or st.session_state.camera:
        if st.session_state.clicked:
                uploaded_file = st.file_uploader("Upload a photo", type=['png', 'jpg', 'jpeg'])
        if st.session_state.camera:
                uploaded_file = st.camera_input("Take a photo")

        if uploaded_file is not None:
                img = PILImage.create(uploaded_file)

                pred, idx, proba = learn_inf.predict(img)
                vocab = learn_inf.dls.vocab
                res_table = pd.DataFrame()
                probas = []

                for res in zip(vocab, proba.tolist()):
                        probas.append(res[1])

                res_table['cloud'] = vocab
                res_table['proba'] = probas
                res_table.nlargest(3, 'proba')

                st.subheader(f'You are looking at `{pred}` clouds!')
                st.write(f'{cloud_desc[pred]}')
                st.link_button("Read more", "https://www.noaa.gov/jetstream/clouds/ten-basic-clouds")
                st.image(img)
                st.write('Other clouds that may be present include:')

                top3 = res_table.nlargest(3, 'proba')
                fig = px.bar(top3, y = 'cloud', x = 'proba', text_auto='.1%', 
                        labels = {'proba':'probability'},
                        color = 'cloud')
                fig.update_traces(textposition="outside", showlegend=False)
                fig.update_xaxes(visible=False)
                st.plotly_chart(fig, use_container_width=True)

                st.write(f'Did the sky change? Click the **upload photo** or **take photo** button again to show us!')
                st.write('*Cloud definitions from **National Oceanic and Atmospheric Administration (NOAA)***')
               
st.divider()
end = '''Inspired by fast.ai's course **Deep Learning for Coders with Fastai and PyTorch**. 
Built on `Fastai`, `PyTorch`, `Streamlit`, and `Plotly`.'''
st.markdown(end)