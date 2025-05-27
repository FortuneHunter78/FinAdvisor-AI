package main

import (
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"inference_service/internal/config"
	"inference_service/internal/delivery/api_delivery"
	"inference_service/internal/delivery/inference_delivery"
	"inference_service/internal/delivery/minio_delivery"
	"inference_service/internal/env"
	"inference_service/internal/usecase/api_usecase"
	"inference_service/internal/usecase/inference_usecase"
	"inference_service/pkg/minio"
	"log/slog"
	"net/http"
)

var (
	envFile = env.InitEnv()
	cfgPath = env.GetString("INFERENCE_CONFIG_PATH", envFile["INFERENCE_CONFIG_PATH"])
)

type application struct {
	srv *http.Server
	cfg *config.Config
}

func createApp() *application {
	cfg, err := config.LoadConfig(cfgPath)
	if err != nil {
		panic(err)
	}

	app := &application{
		cfg: cfg,
	}

	mux := app.mount()

	srv := &http.Server{
		Addr:         app.cfg.HTTPServerConfig.Port,
		Handler:      mux,
		WriteTimeout: app.cfg.HTTPServerConfig.WriteTimeout,
		ReadTimeout:  app.cfg.HTTPServerConfig.ReadTimeout,
		IdleTimeout:  app.cfg.HTTPServerConfig.IdleTimeout,
	}

	app.srv = srv

	return app
}

func (app *application) mount() http.Handler {
	r := chi.NewRouter()

	// Chi middlewares
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	// Minio
	minioClient := minio.NewMinioClient()
	err := minioClient.InitMinio()
	if err != nil {
		slog.Error(err.Error())
	}

	minioHandler := minio_delivery.NewMinioHandler(minioClient)
	minioRouter := minio_delivery.NewMinioRouter(minioHandler)

	// API
	APIUC := api_usecase.NewAPIUC()
	APIHandler := api_delivery.NewAPIHandler(APIUC)
	APIRouter := api_delivery.NewAPIRouter(APIHandler)

	// Inference
	inferenceUC := inference_usecase.NewInferenceUC()
	inferenceHandler := inference_delivery.NewInferenceHandler(inferenceUC)
	inferenceRouter := inference_delivery.NewInferenceRouter(inferenceHandler)

	r.Route("/api", func(r chi.Router) {
		r.Route("/v1", func(r chi.Router) {
			r.Mount("/service", APIRouter)
			r.Mount("/files", minioRouter)
			r.Mount("/inference", inferenceRouter)
		})
	})

	return r
}

func (app *application) run() error {
	slog.Info("[inference]: listening on " + app.cfg.HTTPServerConfig.Host + app.cfg.HTTPServerConfig.Port)
	return app.srv.ListenAndServe()
}
