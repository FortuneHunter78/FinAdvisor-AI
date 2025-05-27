package minio_delivery

import "net/http"

type IMinioHandler interface {
	CreateOne(w http.ResponseWriter, r *http.Request)
	GetOne(w http.ResponseWriter, r *http.Request)
}
