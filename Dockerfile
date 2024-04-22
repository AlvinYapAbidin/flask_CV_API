FROM python:3.9-slim
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Use wget to download your FastSAM model file
# FastSam model
# RUN wget -O FastSAM.pt https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt 
# SAM model (larger size)
RUN wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 

EXPOSE 5000
ENV NAME World
CMD ["python", "run.py"]
