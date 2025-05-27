package config

import "time"

type HTTPServerConfig struct {
	Host         string        `yaml:"host"`
	WriteTimeout time.Duration `yaml:"write_timeout"`
	ReadTimeout  time.Duration `yaml:"read_timeout"`
	IdleTimeout  time.Duration `yaml:"idle_timeout"`
	Port         string        `yaml:"port"`
}
