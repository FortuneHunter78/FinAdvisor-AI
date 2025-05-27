package api_delivery

import (
	"github.com/go-chi/chi/v5"
	"net/http"
)

func NewAPIRouter(h *APIHandler) http.Handler {
	r := chi.NewRouter()

	r.Get("/health", h.HealthCheck)

	return r
}
