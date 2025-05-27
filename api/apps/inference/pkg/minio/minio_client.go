package minio

import (
	"context"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"log/slog"
)

type S3Client struct {
	mc *minio.Client
}

func NewMinioClient() *S3Client {
	return &S3Client{}
}

func (c *S3Client) InitMinio() error {
	ctx := context.Background()

	// Connecting to minio with creds from config
	client, err := minio.New(cfg.S3Config.MinioEndPoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.S3Config.MinioRootUser, cfg.S3Config.MinioRootPassword, ""),
		Secure: cfg.S3Config.MinioUseSSL,
	})
	if err != nil {
		slog.Error("[inference]: Failed to create minio client. Error: " + err.Error())
		return err
	}

	// Establishing minio connection
	c.mc = client

	exists, err := c.mc.BucketExists(ctx, cfg.S3Config.BucketName)
	if err != nil {
		slog.Error("[inference]: Failed to check if bucket exists. Error: " + err.Error())
		return err
	}
	if !exists {
		err := c.mc.MakeBucket(ctx, cfg.S3Config.BucketName, minio.MakeBucketOptions{})
		if err != nil {
			slog.Error("[inference]: Failed to create bucket. Error: " + err.Error())
			return err
		}
	}

	return nil
}
