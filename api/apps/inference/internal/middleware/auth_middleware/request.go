package auth_middleware

import (
	"bytes"
	"encoding/json"
	"inference_service/internal/delivery/http/request_types/service_requests"
	"inference_service/internal/delivery/http/response_types/service_responses"
	"io/ioutil"
	"net/http"
)

func AuthServiceCheckKeyRequest(ch chan service_responses.AuthServiceResponse, authKey string) {
	defer close(ch)

	// Creating request
	var requestBody = service_requests.AuthServiceRequest{
		AuthKey: authKey,
	}

	// Marshalling schema
	marshalledRequest, err := json.Marshal(requestBody)
	if err != nil {
		ch <- service_responses.AuthServiceResponse{
			Status:   400,
			Approved: false,
		}
		return
	}

	// Check if such auth key exists in service
	resp, err := http.Post(authServiceURL, "application/json", bytes.NewBuffer(marshalledRequest))
	if err != nil {
		ch <- service_responses.AuthServiceResponse{
			Status:   400,
			Approved: false,
		}
		return
	}
	defer resp.Body.Close()

	// Validate body
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		ch <- service_responses.AuthServiceResponse{
			Status:   500,
			Approved: false,
		}
		return
	}

	// Unmarshal body
	var authResponse service_responses.AuthServiceResponse
	if err := json.Unmarshal(body, &authResponse); err != nil {
		ch <- service_responses.AuthServiceResponse{
			Status:   500,
			Approved: false,
		}
		return
	}

	// Checking resposne status
	if resp.StatusCode != 200 {
		authResponse.Approved = false
	}

	ch <- authResponse
}
