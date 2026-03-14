# Use official Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app   


# Copy requirements file
COPY requirements.txt .

# Copy the joblib file 
COPY Wine_quality.joblib .
# dot means all the files 

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose port FastAPI will run on
EXPOSE 8000

# Command to run FastAPI with uvicorn
CMD ["uvicorn", "wine_app:app", "--host", "0.0.0.0", "--port", "8000"]
