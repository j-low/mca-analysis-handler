# Use an official Python 3.11 image so pandas installs as a wheel
FROM python:3.11-slim

# install libgomp (OpenMP runtime) so LightGBM can load
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

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
