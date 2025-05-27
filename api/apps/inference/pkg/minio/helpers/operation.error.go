package helpers

type OperationError struct {
	ObjectID string `json:"objectID"`
	Error    error  `json:"error"`
}
