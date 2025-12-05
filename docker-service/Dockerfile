FROM python:3.9-bullseye

WORKDIR /app

RUN apt-get update 

RUN apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements trước để tận dụng cache của Docker
COPY requirements.txt ./


RUN pip install --no-cache-dir -r requirements.txt

COPY model/ /app/model/

COPY . /app/

EXPOSE 5000

CMD ["python", "server.py"]