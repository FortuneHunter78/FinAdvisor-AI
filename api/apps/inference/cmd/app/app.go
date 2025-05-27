package main

import "net/http"

type IApp interface {
	mount() http.Handler
	run() error
}
