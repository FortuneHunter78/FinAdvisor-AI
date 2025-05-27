package auth_middleware

import (
	"inference_service/internal/delivery/http/response_types/service_responses"
	"inference_service/internal/env"
	"net/http"
)

var (
	envFile        = env.InitEnv()
	authServiceURL = env.GetString("AUTH_SERVICE_URL", envFile["AUTH_SERVICE_URL"])
)

func AuthMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			next.ServeHTTP(w, r)
			return
		}

		authKey := r.Header.Get("OAUTH2-KEY")
		if authKey == "" {
			http.Error(w, http.StatusText(http.StatusUnauthorized), http.StatusUnauthorized)
			return
		}

		authChannel := make(chan service_responses.AuthServiceResponse)

		// Requesting in goroutine
		go AuthServiceCheckKeyRequest(authChannel, authKey)

		result := <-authChannel
		if !result.Approved {
			http.Error(w, http.StatusText(http.StatusUnauthorized), http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	})
}
