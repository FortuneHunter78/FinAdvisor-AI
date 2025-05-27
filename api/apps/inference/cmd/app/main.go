package main

import (
	"log/slog"
)

func main() {
	// to hide tensorflow logs
	// export TF_CPP_MIN_LOG_LEVEL=2
	app := createApp()

	go func() {
		slog.Info("[inference]: starting http server...")

		if err := app.run(); err != nil {
			slog.Error("[inference]: Failed to start http server due to error: " + err.Error())
			panic(err)
		}
	}()

	select {}
}
