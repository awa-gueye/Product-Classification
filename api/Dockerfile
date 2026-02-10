# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE $PORT

# Command to run the application
CMD ["uvicorn", "main_render:app", "--host", "0.0.0.0", "--port", "8000"]