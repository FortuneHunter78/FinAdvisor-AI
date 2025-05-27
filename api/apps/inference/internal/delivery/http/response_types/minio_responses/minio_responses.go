package minio_responses

type ErrorResponse struct {
	Error   string      `json:"error"`
	Status  int         `json:"status,omitempty"`
	Details interface{} `json:"details,omitempty"`
}

type SuccessResponse struct {
	Status  int         `json:"status,omitempty"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}
