FROM python:3.12.3-slim

# Install dependencies using pip
RUN pip install pandas==2.2.3 geopandas==1.0.1 streamlit==1.41.1 joblib==1.4.2 scikit-learn==1.6.0

# Set the working directory in the container
WORKDIR /app

# Copy the 'app' directory and its contents into the container's /app folder
COPY app /app

# Set the entry point for the Streamlit application
ENTRYPOINT ["streamlit", "run", "home.py"]
