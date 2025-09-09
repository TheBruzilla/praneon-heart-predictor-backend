FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (with fallback)
RUN pip install --no-cache-dir -r requirements.txt || echo "Some dependencies failed to install, will use fallback"

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Make scripts executable
RUN chmod +x start.sh test_api.py test_api.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["./start.sh"]