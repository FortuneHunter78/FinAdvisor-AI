package inference_delivery

import (
	"github.com/go-chi/chi/v5"
	"net/http"
)

func NewInferenceRouter(h *InferenceHandler) http.Handler {
	r := chi.NewRouter()

	r.Post("/", h.Predict)
	r.Get("/", h.Model)

	return r
}
