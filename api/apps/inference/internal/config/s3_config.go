package config

import "time"

type S3Config struct {
	Port              string `yaml:"port"`
	MinioEndPoint     string `yaml:"minio_endpoint"`
	BucketName        string `yaml:"bucket_name"`
	MinioRootUser     string `yaml:"minio_root_user"`
	MinioRootPassword string `yaml:"minio_root_password"`
	MinioUseSSL       bool   `yaml:"minio_use_ssl"`
	S3VarsConfig      `yaml:"s3_vars"`
}

type S3VarsConfig struct {
	ExpiresIn time.Duration `yaml:"expires_in"`
}
