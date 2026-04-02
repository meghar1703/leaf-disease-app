# Use Python 3.10 (IMPORTANT for TensorFlow compatibility)
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install \
    streamlit==1.22.0 \
    tensorflow==2.10.0 \
    keras==2.10.0 \
    protobuf==3.19.6 \
    numpy==1.23.5 \
    opencv-python-headless \
    pillow \
    h5py

# Expose port (required by Hugging Face)
EXPOSE 7860

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]