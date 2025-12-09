FROM python:3.11-slim

ADD app.py .

COPY requirements.txt .

# Ensure apt lists are updated first
RUN apt-get update -y && \
    # Install the modern replacement packages
    apt-get install -y libgl1 libglx-mesa0 build-essential && \
    # Clean up apt lists to reduce image size
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

CMD ["python", "./app.py"]

