FROM python:3.11

WORKDIR /sd

# install dependencies
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Create a writable cache directory inside the container
RUN mkdir -p /model/cache 

# copy source code
COPY . .

EXPOSE 5000

# Setup an app user so the container doesn't run as the root user
RUN useradd -m sd
USER sd

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
