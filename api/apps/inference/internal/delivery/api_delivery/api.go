package api_delivery

import "net/http"

type IAPIHandler interface {
	HealthCheck(w http.ResponseWriter, r *http.Request)
}
