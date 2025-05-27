package inference_usecase

import (
	"bytes"
	"encoding/json"
	"errors"
	"inference_service/internal/delivery/http/request_types/inference_requests"
	"inference_service/model"
	"io/ioutil"
	"log/slog"
	"net/http"
	"sync"
)

type InferenceUC struct {
	mu *sync.Mutex
	//ml *tg.Model
}

var modelPath = "ml"

func NewInferenceUC() *InferenceUC {
	//mdl := tg.LoadModel(modelPath, []string{"serve"}, nil)
	//if mdl == nil {
	//	return nil
	//}
	//
	//return &InferenceUC{
	//	ml: mdl,
	//}
	return &InferenceUC{}
}

// Predict runs inference on a given prompt
//func (i *InferenceUC) Predict(prompt string) (*tf.Tensor, error) {
//	err := helpers.LoadVocabulary("ml/assets/vocab.txt")
//	if err != nil {
//		slog.Error("[inference]: Failed to read vocab. Error: " + err.Error())
//		return nil, err
//	}
//
//	// Tokenize the input text
//	tokenized, err := helpers.TokenizeText(prompt)
//	if err != nil {
//		slog.Error("[inference]: Error tokenizing text. Error: " + err.Error())
//		return nil, err
//	}
//
//	// Convert tokenized data into Tensors
//	inputWordIds, err := tf.NewTensor(tokenized.InputWordIDs)
//	if err != nil {
//		slog.Error("[inference]: Failed to create inputWordIds. Error: " + err.Error())
//		return nil, err
//	}
//
//	inputMask, err := tf.NewTensor(tokenized.InputMask)
//	if err != nil {
//		slog.Error("[inference]: Failed to create input_mask tensor. Error: " + err.Error())
//		return nil, err
//	}
//
//	inputTypeIds, err := tf.NewTensor(tokenized.InputTypeIDs)
//	if err != nil {
//		slog.Error("[inference]: failed to create input_type_ids tensor. Error: " + err.Error())
//		return nil, err
//	}
//
//	// Execute the ml
//	results := i.ml.Exec([]tf.Output{
//		i.ml.Op("StatefulPartitionedCall", 0), // Adjust based on the ml's output
//	}, map[tf.Output]*tf.Tensor{
//		i.ml.Op("serving_default_input_word_ids", 0): inputWordIds,
//		i.ml.Op("serving_default_input_mask", 0):     inputMask,
//		i.ml.Op("serving_default_input_type_ids", 0): inputTypeIds,
//	})
//
//	// Handle predictions
//	predictions := results[0]
//	return predictions, nil
//}

func (i *InferenceUC) Info() (*model.Model, error) {
	mdl := model.NewModel(modelPath)
	if mdl == nil {
		slog.Error("[inference]: Failed to load ml")
		return nil, errors.New("ml not found")
	}

	return mdl, nil
}

func (i *InferenceUC) Predict(prompt string) (*inference_requests.MLResponse, error) {
	mlRequest := inference_requests.MLRequest{
		Ticker: prompt,
	}

	body, err := json.Marshal(mlRequest)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post("http://ml:8083/predict", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	responseBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var mlResponse inference_requests.MLResponse
	err = json.Unmarshal(responseBody, &mlResponse)
	if err != nil {
		return nil, err
	}

	return &mlResponse, nil
}
