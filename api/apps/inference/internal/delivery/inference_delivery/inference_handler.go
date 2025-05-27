package inference_delivery

import (
	"encoding/json"
	"inference_service/internal/delivery/http/request_types/inference_requests"
	"inference_service/internal/usecase/inference_usecase"
	"net/http"
)

type InferenceHandler struct {
	uc *inference_usecase.InferenceUC
}

func NewInferenceHandler(uc *inference_usecase.InferenceUC) *InferenceHandler {
	return &InferenceHandler{uc: uc}
}

func (h *InferenceHandler) Predict(w http.ResponseWriter, r *http.Request) {
	var request inference_requests.PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	output, err := h.uc.Predict(request.Prompt)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	jsonData, _ := json.Marshal(output)
	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(jsonData)
}

func (h *InferenceHandler) Model(w http.ResponseWriter, r *http.Request) {
	model, err := h.uc.Info()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	jsonData, err := json.Marshal(model)
	if err != nil {
		http.Error(w, "Failed to serialize ml", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(jsonData)
}
