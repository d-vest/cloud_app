### Cloud Classifier App :cloud:

:earth_americas: Web application containing a deep learning model that identifies the type of cloud present based on the image or camera photo provided by the observer.

:brain: The model uses transfer learning by building on a pre-trained convolutional neural network (CNN) model with additional fine-tuning on the task of cloud classification.

:muscle: Training data for fine-tuning was built using [NOAA's Ten Basic Clouds](https://www.noaa.gov/jetstream/clouds/ten-basic-clouds) 
as ground truth for each cloud class. Data was then further expanded by collecting visually similar images for each class using Bing Image Search.  

Built on `Fastai`, `PyTorch`, `Streamlit`, and `Plotly`. Try out the [deployed app](https://cloudapp.streamlit.app)!

Inspired by fast.ai's course **Deep Learning for Coders with Fastai and PyTorch**. 
