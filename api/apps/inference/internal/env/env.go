package env

import (
	"github.com/joho/godotenv"
	"log"
	"os"
	"strconv"
	"time"
)

// InitEnv returns a map with data from .env
func InitEnv() map[string]string {
	envFile, err := godotenv.Read(".env")
	if err != nil {
		log.Fatal(err)
	}

	return envFile
}

// GetDuration is a helper func to work with time.Duration
func GetDuration(stringTime string) (time.Duration, error) {
	duration, err := time.ParseDuration(stringTime)
	if err != nil {
		return 0, err
	}

	return duration, nil
}

// GetString is a helper func to work with .env mapping
func GetString(key string, fallback string) string {
	val, ok := os.LookupEnv(key)
	if !ok {
		return fallback
	}

	return val
}

// GetInt is a helper func to work with .env mapping
func GetInt(key string, fallback int) int {
	val, ok := os.LookupEnv(key)
	if !ok {
		return fallback
	}

	valAsInt, err := strconv.Atoi(val)
	if err != nil {
		return fallback
	}

	return valAsInt
}
