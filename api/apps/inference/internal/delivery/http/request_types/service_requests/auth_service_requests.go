package service_requests

// AuthServiceRequest To validate if auth service has this user key
type AuthServiceRequest struct {
	AuthKey string `json:"auth_key"`
}
