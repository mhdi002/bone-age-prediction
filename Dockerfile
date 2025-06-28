# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model files into the container at /app
COPY bone_age_gui.py /app/
COPY train.py /app/ # Although train.py might not be needed for the GUI, including it for completeness
COPY best_model_fold0.pth /app/
COPY best_model_fold1.pth /app/
COPY best_model_fold2.pth /app/
COPY best_model_fold3.pth /app/
COPY best_model_fold4.pth /app/

# Make port 80 available to the world outside this container (Gradio default port is 7860)
EXPOSE 7860

# Define environment variable
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run bone_age_gui.py when the container launches
CMD ["python", "bone_age_gui.py"]
