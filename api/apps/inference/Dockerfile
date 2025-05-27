FROM golang:1.24 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download
COPY . .

RUN go mod tidy
#
#RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz && \
#    tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz && \
#    ldconfig && \
#    rm libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz
#
#RUN wget https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.4.0.tar.gz && \
#    tar -C /usr/local -xzf v1.4.0.tar.gz && \
#    ldconfig && \
#    rm v1.4.0.tar.gz

RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -o app ./cmd/app/*.go
RUN chmod +x /app/app

FROM debian:bookworm-slim AS final

WORKDIR /app

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
#
#COPY --from=builder /usr/local/lib/libtensorflow.so.2 /usr/local/lib/
#COPY --from=builder /usr/local/lib/libtensorflow_framework.so.2 /usr/local/lib/

# RUN ldconfig

COPY --from=builder /app/app /app/app
RUN chmod +x /app/app

COPY .env /app/.env
COPY .minio.env /app/.minio.env
COPY model /app/model

RUN mkdir -p /app/config
COPY --from=builder /app/config/prod.yml /app/config/prod.yml
#
#COPY ./model/libonnxruntime.so /usr/local/lib
#COPY ./model/onnxruntime.so /usr/local/lib

CMD ["./app"]
