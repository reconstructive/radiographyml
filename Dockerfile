#FROM python:3.6-slim-stretch
FROM python:3.7.7-stretch
#FROM python:3.7.7-alpine3.11

ADD requirements.txt /
RUN pip install -r /requirements.txt

ADD . /app
WORKDIR /app
#COPY . .

EXPOSE 5000
CMD [ "python" , "app.py"]
