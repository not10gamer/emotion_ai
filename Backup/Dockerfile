# Stage 1: Build stage to install dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build-time system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages for a specific user
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------
# Stage 2: Final production stage
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage to the appuser's home
COPY --from=builder /root/.local /home/appuser/.local

# Copy the application code
COPY . .

# Fix permissions for the appuser to prevent PermissionErrors
# This allows deepface to create its .deepface cache directory
RUN chown -R appuser:appuser /app /home/appuser

# Switch to the non-root user
USER appuser

# Add the user's local bin to the PATH so python can find packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Now run preloading as the appuser, who can find the packages and has permissions
RUN python preload_models.py

# Set environment variables for the running container
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# Use exec form for better signal handling
CMD ["sh", "-c", "exec gunicorn --bind :$PORT --workers 2 --threads 2 --timeout 300 --access-logfile - --error-logfile - enhanced_backend:app"]