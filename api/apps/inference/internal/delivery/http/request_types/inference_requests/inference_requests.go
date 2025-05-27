package inference_requests

type PredictRequest struct {
	Prompt string `json:"prompt"`
}

type MLRequest struct {
	Ticker string `json:"ticker"`
}

type MLResponse struct {
	Output []float64 `json:"output"`
}
