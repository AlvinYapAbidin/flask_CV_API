FROM python:3.9-slim
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Use wget to download your FastSAM model file
RUN wget -O FastSAM.pt https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt

EXPOSE 5000
ENV NAME World
CMD ["python", "run.py"]
