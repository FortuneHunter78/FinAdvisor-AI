package inference_delivery

import "net/http"

type IInferenceHandler interface {
	Predict(w http.ResponseWriter, r *http.Request)
	Model(w http.ResponseWriter, r *http.Request)
}
