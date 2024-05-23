FROM python:3.9.13
ENV PYTHONUNBUFFERED 1
RUN mkdir /web_flask
WORKDIR /web_flask
COPY requirements.txt /web_flask/
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install -U Flask-SQLAlchemy
RUN pip install -U pymysql
RUN pip install -U cryptography
RUN pip install -U flask-migrate
RUN pip install -U matplotlib
RUN pip install -U Flask-Plots
RUN pip install -U pandas
RUN pip install -U tensorflow==2.9.3
ADD . /web_flask/