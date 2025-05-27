package model

import (
	"inference_service/internal/usecase/inference_usecase/helpers"
)

type Model struct {
	Path    string                  `json:"path"`
	Version string                  `json:"version"`
	Input   helpers.TokenizedOutput `json:"input"`
	Output  [][]int32               `json:"output"`
}

func NewModel(path string) *Model {
	return &Model{
		Path:    path,
		Version: "1.0.0",
		Input: helpers.TokenizedOutput{
			InputMask:    [][]int32{{1, 2}, {3, 4}},
			InputTypeIDs: [][]int32{{1, 2}, {3, 4}},
			InputWordIDs: [][]int32{{1, 2}, {3, 4}},
		},
		Output: [][]int32{{1, 2}, {3, 4}},
	}
}
