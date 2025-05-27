package helpers

type FileDataType struct {
	FileName string `json:"filename"`
	Data     []byte `json:"data"`
}
