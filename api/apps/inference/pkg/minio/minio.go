package minio

import "inference_service/pkg/minio/helpers"

type Client interface {
	InitMinio() error
	CreateOne(file helpers.FileDataType) (string, error)
	GetOne(objectID string) (string, error)
}
