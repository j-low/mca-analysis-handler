# Use an official Python 3.11 image so pandas installs as a wheel
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Copy in just the requirements first (for layer caching)
COPY requirements.txt .

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port if needed (e.g. for gunicorn)
EXPOSE 8000

# Start the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers=1"]
