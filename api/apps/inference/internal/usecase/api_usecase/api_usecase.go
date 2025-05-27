package api_usecase

import (
	"sync"
)

type APIUC struct {
	mu *sync.Mutex
}

func NewAPIUC() *APIUC {
	return &APIUC{}
}
