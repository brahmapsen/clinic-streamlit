# Streamlit Dockerfile (optional - for containerized deployment)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements-streamlit.txt .
RUN pip install --no-cache-dir -r requirements-streamlit.txt

# Copy Streamlit app
COPY streamlit_ui.py .
COPY .env* ./

# Expose Streamlit port
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "streamlit_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
