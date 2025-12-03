#Using official lightweight Python image
FROM python:3.11-slim

#Set the working directory in the comtainer
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    python3-dev \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*



#Copy the dependencies file to the working directory

COPY ml-app/requirements.txt .

#Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


#Copy the app code into container
#This assumes the docker build context is the root of the project


COPY ml-app/entrypoint ./entrypoint/
COPY ml-app/src ./src/
COPY common ./common/


#Run if not using DCF
COPY config/ /config/
COPY data/ /data/
COPY models/ /models/


ENV PYTHONPATH="/app:/common:/app/src"

CMD ["python", "entrypoint/train.py"]
# CMD ["python", "entrypoint/inference_api.py"]



