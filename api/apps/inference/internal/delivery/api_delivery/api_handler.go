package api_delivery

import (
	"encoding/json"
	"inference_service/internal/delivery/http/response_types/api_responses"
	"inference_service/internal/usecase/api_usecase"
	"net/http"
)

type APIHandler struct {
	uc *api_usecase.APIUC
}

func NewAPIHandler(uc *api_usecase.APIUC) *APIHandler {
	return &APIHandler{uc: uc}
}

func (h *APIHandler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	message := api_responses.APIHealthCheckResponse{
		Message: "It's working",
		Status:  "200",
	}

	if err := json.NewEncoder(w).Encode(&message); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}
