# Use an official Python runtime as a parent image
FROM python:3.13

# Set the working directory in the container
WORKDIR /app

# Copy the chatbot code into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the necessary port (adjust based on your chatbot framework)
EXPOSE 8501

# Run the chatbot application

CMD ["streamlit", "run", "main3.py", "--server.port=8501", "--server.address=0.0.0.0"]
