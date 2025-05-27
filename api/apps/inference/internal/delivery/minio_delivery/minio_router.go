package minio_delivery

import (
	"github.com/go-chi/chi/v5"
	"net/http"
)

func NewMinioRouter(h *MinioHandler) http.Handler {
	r := chi.NewRouter()

	r.Post("/", h.CreateOne)

	r.Get("/{object_id}", h.GetOne)

	return r
}
