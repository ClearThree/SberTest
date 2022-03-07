FROM mfeurer/auto-sklearn:master

COPY ./requirements.txt /automl/requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir --upgrade -r /automl/requirements.txt

COPY . /automl
WORKDIR /automl
CMD ["python3", "./showcase.py"]