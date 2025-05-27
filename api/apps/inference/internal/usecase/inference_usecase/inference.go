package inference_usecase

import (
	"inference_service/internal/delivery/http/request_types/inference_requests"
	"inference_service/model"
)

type IInferenceUC interface {
	Predict(prompt string) (*inference_requests.MLResponse, error)
	Info() (*model.Model, error)
}
