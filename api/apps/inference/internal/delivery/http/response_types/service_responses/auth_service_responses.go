package service_responses

type AuthServiceResponse struct {
	Approved bool `json:"approved"`
	Status   int  `json:"status"`
}
