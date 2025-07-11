# Base Image
FROM python:3.12.4-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --requirement requirements.txt --no-cache-dir

COPY . .

# Port
EXPOSE 8080

# ENTRYPOINT
ENTRYPOINT ["bash"]