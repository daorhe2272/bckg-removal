FROM python:3.9-slim as base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Test stage
FROM base as test

COPY src/ ./src/
COPY tests/ ./tests/
COPY pytest.ini .
COPY run_tests.py .
COPY models/ ./models/

ENTRYPOINT ["python", "run_tests.py"]

# Production stage
FROM base as production

COPY src/app.py .
COPY src/.streamlit/ ./.streamlit/
COPY models/ ./models/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 