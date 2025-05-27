package config

import (
	"gopkg.in/yaml.v3"
	"io/ioutil"
	"log/slog"
	"os"
)

type Config struct {
	HTTPServerConfig `yaml:"http_server"`
	S3Config         `yaml:"s3"`
	RuntimeConfig    `yaml:"runtime"`
}

func LoadConfig(path string) (*Config, error) {
	file, err := os.Open(path)
	if err != nil {
		slog.Error("[inference]: failed to open config file. Error: ", err.Error())
		return nil, err
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		slog.Error("[inference]: failed to read config file. Error: ", err.Error())
		return nil, err
	}

	var config Config
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		slog.Error("[inference]: failed to parse config file. Error: ", err.Error())
		return nil, err
	}

	return &config, nil
}
