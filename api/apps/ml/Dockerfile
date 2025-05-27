FROM python:3.12-slim

WORKDIR /src

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .

RUN pip install --timeout=10000 --no-cache-dir --upgrade -r requirements.txt

COPY . .

COPY ./src/model/model.onnx /src/model/model.onnx
EXPOSE 8083

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8083", "--reload"]
