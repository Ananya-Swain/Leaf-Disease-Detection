import streamlit as st
import tensorflow as tf
import numpy as np
from disease_tips import disease_tips
from PIL import Image
import base64
import requests

# Setting api keys
SERPER_API_KEY = "e5182fa68c1a062e2b2f257b24d857e854cb34db"
TOGETHER_API_KEY = "c42cc6fb2dd14c4537f69407d966e489e46da62a0a097c5fd437fb83bc954ce8"

#Define Class
class_name = ['Apple___Apple_scab',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___healthy',
 'Potato___Early_blight',
 'Potato___healthy',
 'Tomato___Early_blight',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___healthy']


#Tensorflow Model Prediction
def model_prediction(test_image) :
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


#Code for creating chatbot
def google_search(query):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"q": query}
    response = requests.post(url, headers=headers, json=data)
    return response.json()


def summarize_with_together(text):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Free model on Together.ai
        "messages": [
            {"role": "user", "content": f"Summarize the following:\n\n{text}"}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']

# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #e0ff33;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.markdown(
#     """
#     <style>
#     section[data-testid="stSidebar"] {
#         background-color: #868487;
#     }
#     .stApp {
#         background-image: url("https://images.unsplash.com/photo-1517816743773-6e0fd518b4a6");
#         background-size: cover;
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Set solid background color for sidebar only
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background-color: #e6ffe6; /* Light green */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def set_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #b3dc9c;
            background-size: cover;
        }}
        
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
set_bg_from_local("indoor-plants-studio.jpg")

# Inject CSS to style buttons
st.markdown(
    """
    <style>
    /* Button background and text color */
    div.stButton > button {
        background-color: #4CAF50;  /* Green */
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 8px;
        font-size: 16px;
        transition: background-color 0.3s;
    }

    /* Hover effect */
    div.stButton > button:hover {
        background-color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# st.sidebar.title("Dashboard")
st.sidebar.markdown(
    "<h3 style='color: black;font-size: 28px'>Dashboard</h3>",
    unsafe_allow_html=True
)


# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"
    
# Sidebar navigation without radio buttons
if st.sidebar.button("Home"):
    st.session_state.page = "Home"
if st.sidebar.button("About"):
    st.session_state.page = "About"
if st.sidebar.button("Disease Recognition"):
    st.session_state.page = "Disease Recognition"
# if st.sidebar.button("Ask AI"):
#     st.session_state.page = "Ask AI"

#Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.radio("",['Home','About','Disease Recognition'])

#Home Page
# if(app_mode == 'Home') :
if st.session_state.page == "Home" :
    st.header("CURIFY")
    image_path = "./home_page.jpeg"
    # image = Image.open(image_path)
    # resized = image.resize((300, 200))
    # col1, col2, col3 = st.columns([1, 2, 1])
    # with col2:
    st.image(image_path) #use_column_width = True
    st.markdown("""
    Welcome to the Curify! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
""")
    
#About Page
# elif(app_mode == 'About') :
elif st.session_state.page == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
    This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. train (19085 images)
    2. valid (4745 images)
    3. test (33 images)

""")


#Prediction Page
# elif(app_mode == 'Disease Recognition') :
elif st.session_state.page == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload image of a single leaf.")

    if(st.button("Show Image")) :
        if test_image :
            st.image(test_image) #use_column_width = True
        else :
            st.error("Please provide the image first.")
        

    #Predict Button
    if(st.button("Predict")) :
        #with st.spinner("Please Wait...") :
        if test_image :
            st.write("Predicting")
            result_index = model_prediction(test_image)
            predicted_class = class_name[result_index]
        
            # Save to session state
            st.session_state["predicted_class"] = predicted_class
            st.success("It's a {}.".format(predicted_class))
        else :
            st.error("Please provide the image first.")

    #Tips Button
        # if(st.button("Get Curing Tips")) :
        #     if predicted_class :
        #         st.write(predicted_class)
        #         tips = disease_tips.get(predicted_class)
        #         if tips :
        #             st.success(f"Curing tips for **{predicted_class.title()}**:")
        #             st.write(tips)
        #         else :
        #             st.error("Disease not found in the database. Please try another name.")
        #     else :
        #         st.error("User Name is not defined.")

    if st.button("Get Curing Tips"):
        if "predicted_class" in st.session_state:
            user_input = st.session_state["predicted_class"]
            tips = disease_tips.get(user_input)
            if tips:
                #st.success(f"Curing tips for **{user_input.title()}**:")
                st.markdown(tips)
            else:
                st.error("No tips found for this disease.")
        else:
            st.error("Please run the prediction first.")

#Ask AI button
elif st.session_state.page == "Ask AI" :
    st.header("Ask AI")
    st.title("Ask Anything about your plants")
    question = st.text_input("What's your question?")
    if question:
        with st.spinner("Searching..."):
            results = google_search(question)
        if 'organic' in results:
            snippets = "\n\n".join([item['snippet'] for item in results['organic'][:5]])
            with st.spinner("Summarizing with free AI..."):
                summary = summarize_with_together(snippets)

            st.subheader("Answer:")
            st.write(summary)

            with st.expander("Sources"):
                for item in results['organic'][:5]:
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                    st.write(item['snippet'])
                    st.markdown("---")
        else:
            st.error("No results found.")