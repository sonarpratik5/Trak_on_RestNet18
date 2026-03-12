FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Where everything lives inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create a non-root user and give permissions
RUN useradd -m app && chown -R app:app /app
USER app

# Run your script
CMD ["python", "scripts/resnet_trak.py"]
