package minio

import (
	"bytes"
	"context"
	"github.com/google/uuid"
	"github.com/minio/minio-go/v7"
	"inference_service/internal/config"
	"inference_service/internal/env"
	"inference_service/pkg/minio/helpers"
	"log/slog"
)

var (
	envFile = env.InitEnv()
	cfgPath = env.GetString("INFERENCE_CONFIG_PATH", envFile["INFERENCE_CONFIG_PATH"])
	cfg, _  = config.LoadConfig(cfgPath)
)

func (c *S3Client) CreateOne(file helpers.FileDataType) (string, error) {
	// Generating unique id
	objectID := uuid.New().String()

	// Stream for reading bytes from minio
	reader := bytes.NewReader(file.Data)

	// Publishing data into minio
	_, err := c.mc.PutObject(context.Background(), cfg.S3Config.BucketName,
		objectID, reader,
		int64(len(file.Data)), minio.PutObjectOptions{})

	if err != nil {
		slog.Error("[inference]: Failed to create entity in minio")
		return "", err
	}

	// Getting url of created back
	_, err = c.mc.PresignedGetObject(context.Background(), cfg.S3Config.BucketName,
		objectID, cfg.S3Config.S3VarsConfig.ExpiresIn,
		nil)
	if err != nil {
		slog.Error("[inference]: Failed to pull entity from minio. Error: " + err.Error())
		return "", err
	}

	return objectID, nil
}

func (c *S3Client) GetOne(objectID string) (string, error) {
	url, err := c.mc.PresignedGetObject(context.Background(), cfg.S3Config.BucketName,
		objectID, cfg.S3Config.S3VarsConfig.ExpiresIn,
		nil)
	if err != nil {
		slog.Error("[inference]: Failed to pull entity from minio. Error: " + err.Error())
		return "", err
	}

	return url.String(), nil
}
