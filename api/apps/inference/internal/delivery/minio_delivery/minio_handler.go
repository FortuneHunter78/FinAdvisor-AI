package minio_delivery

import (
	"encoding/json"
	"github.com/go-chi/chi/v5"
	"inference_service/internal/delivery/http/response_types/minio_responses"
	"inference_service/pkg/minio"
	"inference_service/pkg/minio/helpers"
	"io"
	"net/http"
)

type MinioHandler struct {
	client *minio.S3Client
}

func NewMinioHandler(client *minio.S3Client) *MinioHandler {
	return &MinioHandler{client: client}
}

func (h *MinioHandler) CreateOne(w http.ResponseWriter, r *http.Request) {
	err := r.ParseMultipartForm(10 << 20) // 10 MB max file size
	if err != nil {
		http.Error(w, "Не удалось прочитать файл", http.StatusBadRequest)
		return
	}

	file, handler, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Не удалось получить файл", http.StatusBadRequest)
		return
	}
	defer file.Close()

	fileBytes, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "Не удалось прочитать содержимое файла", http.StatusInternalServerError)
		return
	}

	// Создаем структуру FileDataType для хранения данных файла
	fileData := helpers.FileDataType{
		FileName: handler.Filename,
		Data:     fileBytes,
	}

	// Сохраняем файл в MinIO с помощью метода CreateOne
	link, err := h.client.CreateOne(fileData)
	if err != nil {
		http.Error(w, "Не удалось сохранить файл", http.StatusInternalServerError)
		return
	}

	// Возвращаем успешный ответ с URL-адресом сохраненного файла
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	successResponse := minio_responses.SuccessResponse{
		Status:  200,
		Message: "File upload success",
		Data:    link,
	}
	jsonResponse, err := json.Marshal(successResponse)
	w.Write(jsonResponse)
}

func (h *MinioHandler) GetOne(w http.ResponseWriter, r *http.Request) {
	objectID := chi.URLParam(r, "object_id")

	link, err := h.client.GetOne(objectID)
	if err != nil {
		errorResponse := minio_responses.ErrorResponse{
			Status:  http.StatusInternalServerError,
			Error:   "Unable to get the object",
			Details: err,
		}
		jsonResponse, _ := json.Marshal(errorResponse)
		w.Write(jsonResponse)
		return
	}

	succeesResponse := minio_responses.SuccessResponse{
		Status:  200,
		Message: "File download success",
		Data:    link,
	}
	jsonResponse, _ := json.Marshal(succeesResponse)
	w.Write(jsonResponse)
}
