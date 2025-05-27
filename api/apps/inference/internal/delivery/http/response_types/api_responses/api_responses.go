package api_responses

type APIHealthCheckResponse struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}
