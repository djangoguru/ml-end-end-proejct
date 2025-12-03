#Using official lightweight Python image
FROM python:3.11-slim



#Set global enviroment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH: /app

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

COPY ui-app/requirements.txt .

#Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


#Copy the app code into container
#This assumes the docker build context is the root of the project


COPY ui-app/ .
COPY common ./common/
COPY ml-app/src ./ml-app/src/




EXPOSE 8050


CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050"]
# CMD ["python", "entrypoint/inference_api.py"]



