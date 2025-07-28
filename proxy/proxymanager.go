package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	PROFILE_SPLIT_CHAR = ":"
)

type ProxyManager struct {
	sync.Mutex

	config    Config
	ginEngine *gin.Engine

	ollamaTranslator *OllamaToOAITranslator

	// logging
	proxyLogger    *LogMonitor
	upstreamLogger *LogMonitor
	muxLogger      *LogMonitor

	metricsMonitor *MetricsMonitor

	processGroups map[string]*ProcessGroup

	// shutdown signaling
	shutdownCtx    context.Context
	shutdownCancel context.CancelFunc
}

func New(config Config) *ProxyManager {
	// set up loggers
	stdoutLogger := NewLogMonitorWriter(os.Stdout)
	upstreamLogger := NewLogMonitorWriter(stdoutLogger)
	proxyLogger := NewLogMonitorWriter(stdoutLogger)

	if config.LogRequests {
		proxyLogger.Warn("LogRequests configuration is deprecated. Use logLevel instead.")
	}

	switch strings.ToLower(strings.TrimSpace(config.LogLevel)) {
	case "debug":
		proxyLogger.SetLogLevel(LevelDebug)
		upstreamLogger.SetLogLevel(LevelDebug)
	case "info":
		proxyLogger.SetLogLevel(LevelInfo)
		upstreamLogger.SetLogLevel(LevelInfo)
	case "warn":
		proxyLogger.SetLogLevel(LevelWarn)
		upstreamLogger.SetLogLevel(LevelWarn)
	case "error":
		proxyLogger.SetLogLevel(LevelError)
		upstreamLogger.SetLogLevel(LevelError)
	default:
		proxyLogger.SetLogLevel(LevelInfo)
		upstreamLogger.SetLogLevel(LevelInfo)
	}

	shutdownCtx, shutdownCancel := context.WithCancel(context.Background())

	pm := &ProxyManager{
		config:    config,
		ginEngine: gin.New(),

		proxyLogger:    proxyLogger,
		muxLogger:      stdoutLogger,
		upstreamLogger: upstreamLogger,

		metricsMonitor: NewMetricsMonitor(&config),

		processGroups: make(map[string]*ProcessGroup),

		shutdownCtx:    shutdownCtx,
		shutdownCancel: shutdownCancel,
	}

	pm.ollamaTranslator = &OllamaToOAITranslator{pm: pm}

	// create the process groups
	for groupID := range config.Groups {
		processGroup := NewProcessGroup(groupID, config, proxyLogger, upstreamLogger)
		pm.processGroups[groupID] = processGroup
	}

	pm.setupGinEngine()
	return pm
}

func (pm *ProxyManager) setupGinEngine() {
	pm.ginEngine.Use(func(c *gin.Context) {
		// Start timer
		start := time.Now()

		// capture these because /upstream/:model rewrites them in c.Next()
		clientIP := c.ClientIP()
		method := c.Request.Method
		path := c.Request.URL.Path

		// Process request
		c.Next()

		// Stop timer
		duration := time.Since(start)

		statusCode := c.Writer.Status()
		bodySize := c.Writer.Size()

		pm.proxyLogger.Infof("Request %s \"%s %s %s\" %d %d \"%s\" %v",
			clientIP,
			method,
			path,
			c.Request.Proto,
			statusCode,
			bodySize,
			c.Request.UserAgent(),
			duration,
		)
	})

	// see: issue: #81, #77 and #42 for CORS issues
	// respond with permissive OPTIONS for any endpoint
	pm.ginEngine.Use(func(c *gin.Context) {
		if c.Request.Method == "OPTIONS" {
			c.Header("Access-Control-Allow-Origin", "*")
			c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")

			// allow whatever the client requested by default
			if headers := c.Request.Header.Get("Access-Control-Request-Headers"); headers != "" {
				sanitized := SanitizeAccessControlRequestHeaderValues(headers)
				c.Header("Access-Control-Allow-Headers", sanitized)
			} else {
				c.Header(
					"Access-Control-Allow-Headers",
					"Content-Type, Authorization, Accept, X-Requested-With",
				)
			}
			c.Header("Access-Control-Max-Age", "86400")
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	})

	mm := MetricsMiddleware(pm)
	// Ollama API passthrough routes
	pm.ginEngine.POST("/api/chat", pm.proxyOllamaPassthrough)
	pm.ginEngine.POST("/api/generate", pm.proxyOllamaPassthrough)
	pm.ginEngine.POST("/api/embed", pm.proxyOllamaPassthrough)
	pm.ginEngine.GET("/api/tags", pm.handleOllamaTagsRequest)
	pm.ginEngine.GET("/api/ps", pm.handleOllamaPsRequest)
	pm.ginEngine.POST("/api/show", pm.handleOllamaShowRequest)

	// Set up routes using the Gin engine
	pm.ginEngine.POST("/v1/chat/completions", mm, pm.proxyOAIHandler)
	// Support legacy /v1/completions api, see issue #12
	pm.ginEngine.POST("/v1/completions", mm, pm.proxyOAIHandler)

	// Support embeddings
	pm.ginEngine.POST("/v1/embeddings", mm, pm.proxyOAIHandler)
	pm.ginEngine.POST("/v1/rerank", mm, pm.proxyOAIHandler)
	pm.ginEngine.POST("/v1/reranking", mm, pm.proxyOAIHandler)
	pm.ginEngine.POST("/rerank", mm, pm.proxyOAIHandler)

	// Support audio/speech endpoint
	pm.ginEngine.POST("/v1/audio/speech", pm.proxyOAIHandler)
	pm.ginEngine.POST("/v1/audio/transcriptions", pm.proxyOAIPostFormHandler)

	pm.ginEngine.GET("/v1/models", pm.listModelsHandler)

	// in proxymanager_loghandlers.go
	pm.ginEngine.GET("/logs", pm.sendLogsHandlers)
	pm.ginEngine.GET("/logs/stream", pm.streamLogsHandler)
	pm.ginEngine.GET("/logs/stream/:logMonitorID", pm.streamLogsHandler)

	/**
	 * User Interface Endpoints
	 */
	pm.ginEngine.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusFound, "/ui")
	})

	pm.ginEngine.GET("/upstream", func(c *gin.Context) {
		c.Redirect(http.StatusFound, "/ui/models")
	})
	pm.ginEngine.Any("/upstream/:model_id/*upstreamPath", pm.proxyToUpstream)

	pm.ginEngine.GET("/unload", pm.unloadAllModelsHandler)
	pm.ginEngine.GET("/running", pm.listRunningProcessesHandler)

	pm.ginEngine.GET("/favicon.ico", func(c *gin.Context) {
		if data, err := reactStaticFS.ReadFile("ui_dist/favicon.ico"); err == nil {
			c.Data(http.StatusOK, "image/x-icon", data)
		} else {
			c.String(http.StatusInternalServerError, err.Error())
		}
	})

	reactFS, err := GetReactFS()
	if err != nil {
		pm.proxyLogger.Errorf("Failed to load React filesystem: %v", err)
	} else {

		// serve files that exist under /ui/*
		pm.ginEngine.StaticFS("/ui", reactFS)

		// server SPA for UI under /ui/*
		pm.ginEngine.NoRoute(func(c *gin.Context) {
			if !strings.HasPrefix(c.Request.URL.Path, "/ui") {
				c.AbortWithStatus(http.StatusNotFound)
				return
			}

			file, err := reactFS.Open("index.html")
			if err != nil {
				c.String(http.StatusInternalServerError, err.Error())
				return
			}
			defer file.Close()
			http.ServeContent(c.Writer, c.Request, "index.html", time.Now(), file)

		})
	}

	// see: proxymanager_api.go
	// add API handler functions
	addApiHandlers(pm)

	// Disable console color for testing
	gin.DisableConsoleColor()
}

// ServeHTTP implements http.Handler interface
func (pm *ProxyManager) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	pm.ginEngine.ServeHTTP(w, r)
}

// StopProcesses acquires a lock and stops all running upstream processes.
// This is the public method safe for concurrent calls.
// Unlike Shutdown, this method only stops the processes but doesn't perform
// a complete shutdown, allowing for process replacement without full termination.
func (pm *ProxyManager) StopProcesses(strategy StopStrategy) {
	pm.Lock()
	defer pm.Unlock()

	// stop Processes in parallel
	var wg sync.WaitGroup
	for _, processGroup := range pm.processGroups {
		wg.Add(1)
		go func(processGroup *ProcessGroup) {
			defer wg.Done()
			processGroup.StopProcesses(strategy)
		}(processGroup)
	}

	wg.Wait()
}

// Shutdown stops all processes managed by this ProxyManager
func (pm *ProxyManager) Shutdown() {
	pm.Lock()
	defer pm.Unlock()

	pm.proxyLogger.Debug("Shutdown() called in proxy manager")

	var wg sync.WaitGroup
	// Send shutdown signal to all process in groups
	for _, processGroup := range pm.processGroups {
		wg.Add(1)
		go func(processGroup *ProcessGroup) {
			defer wg.Done()
			processGroup.Shutdown()
		}(processGroup)
	}
	wg.Wait()
	pm.shutdownCancel()
}

func (pm *ProxyManager) swapProcessGroup(requestedModel string) (*ProcessGroup, string, error) {
	// de-alias the real model name and get a real one
	realModelName, found := pm.config.RealModelName(requestedModel)
	if !found {
		return nil, realModelName, fmt.Errorf("could not find real modelID for %s", requestedModel)
	}

	processGroup := pm.findGroupByModelName(realModelName)
	if processGroup == nil {
		return nil, realModelName, fmt.Errorf("could not find process group for model %s", requestedModel)
	}

	if processGroup.exclusive {
		pm.proxyLogger.Debugf("Exclusive mode for group %s, stopping other process groups", processGroup.id)
		for groupId, otherGroup := range pm.processGroups {
			if groupId != processGroup.id && !otherGroup.persistent {
				// Check if this is an Ollama group
				if otherGroup.backendType == "ollama" {
					pm.proxyLogger.Debugf("Unloading Ollama group: %s", groupId)
					go pm.sendOllamaUnloadAll(otherGroup)
				} else {
					// Regular process group - stop processes
					otherGroup.StopProcesses(StopWaitForInflightRequest)
				}
			}
		}
	}

	return processGroup, realModelName, nil
}

func (pm *ProxyManager) listModelsHandler(c *gin.Context) {
	data := make([]gin.H, 0, len(pm.config.Models))
	createdTime := time.Now().Unix()

	for id, modelConfig := range pm.config.Models {
		if modelConfig.Unlisted {
			continue
		}

		record := gin.H{
			"id":       id,
			"object":   "model",
			"created":  createdTime,
			"owned_by": "llama-swap",
		}

		if name := strings.TrimSpace(modelConfig.Name); name != "" {
			record["name"] = name
		}
		if desc := strings.TrimSpace(modelConfig.Description); desc != "" {
			record["description"] = desc
		}

		data = append(data, record)
	}

	// Set CORS headers if origin exists
	if origin := c.GetHeader("Origin"); origin != "" {
		c.Header("Access-Control-Allow-Origin", origin)
	}

	// Use gin's JSON method which handles content-type and encoding
	c.JSON(http.StatusOK, gin.H{
		"object": "list",
		"data":   data,
	})
}

func (pm *ProxyManager) proxyToUpstream(c *gin.Context) {
	requestedModel := c.Param("model_id")

	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "model id required in path")
		return
	}

	processGroup, _, err := pm.swapProcessGroup(requestedModel)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error swapping process group: %s", err.Error()))
		return
	}

	// rewrite the path
	c.Request.URL.Path = c.Param("upstreamPath")
	processGroup.ProxyRequest(requestedModel, c.Writer, c.Request)
}

func (pm *ProxyManager) proxyOAIHandler(c *gin.Context) {
	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "could not ready request body")
		return
	}

	requestedModel := gjson.GetBytes(bodyBytes, "model").String()
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "missing or invalid 'model' key")
		return
	}

	realModelName, found := pm.config.RealModelName(requestedModel)
	if !found {
		pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("could not find real modelID for %s", requestedModel))
		return
	}

	processGroup, _, err := pm.swapProcessGroup(realModelName)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error swapping process group: %s", err.Error()))
		return
	}

	// issue #69 allow custom model names to be sent to upstream
	useModelName := pm.config.Models[realModelName].UseModelName
	if useModelName != "" {
		bodyBytes, err = sjson.SetBytes(bodyBytes, "model", useModelName)
		if err != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error rewriting model name in JSON: %s", err.Error()))
			return
		}
	}

	// issue #174 strip parameters from the JSON body
	stripParams, err := pm.config.Models[realModelName].Filters.SanitizedStripParams()
	if err != nil { // just log it and continue
		pm.proxyLogger.Errorf("Error sanitizing strip params string: %s, %s", pm.config.Models[realModelName].Filters.StripParams, err.Error())
	} else {
		for _, param := range stripParams {
			pm.proxyLogger.Debugf("<%s> stripping param: %s", realModelName, param)
			bodyBytes, err = sjson.DeleteBytes(bodyBytes, param)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error deleting parameter %s from request", param))
				return
			}
		}
	}

	c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	// dechunk it as we already have all the body bytes see issue #11
	c.Request.Header.Del("transfer-encoding")
	c.Request.Header.Set("content-length", strconv.Itoa(len(bodyBytes)))
	c.Request.ContentLength = int64(len(bodyBytes))

	if err := processGroup.ProxyRequest(realModelName, c.Writer, c.Request); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
		pm.proxyLogger.Errorf("Error Proxying Request for processGroup %s and model %s", processGroup.id, realModelName)
		return
	}
}

func (pm *ProxyManager) proxyOAIPostFormHandler(c *gin.Context) {
	// Parse multipart form
	if err := c.Request.ParseMultipartForm(32 << 20); err != nil { // 32MB max memory, larger files go to tmp disk
		pm.sendErrorResponse(c, http.StatusBadRequest, fmt.Sprintf("error parsing multipart form: %s", err.Error()))
		return
	}

	// Get model parameter from the form
	requestedModel := c.Request.FormValue("model")
	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "missing or invalid 'model' parameter in form data")
		return
	}

	processGroup, realModelName, err := pm.swapProcessGroup(requestedModel)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error swapping process group: %s", err.Error()))
		return
	}

	// We need to reconstruct the multipart form in any case since the body is consumed
	// Create a new buffer for the reconstructed request
	var requestBuffer bytes.Buffer
	multipartWriter := multipart.NewWriter(&requestBuffer)

	// Copy all form values
	for key, values := range c.Request.MultipartForm.Value {
		for _, value := range values {
			fieldValue := value
			// If this is the model field and we have a profile, use just the model name
			if key == "model" {
				// # issue #69 allow custom model names to be sent to upstream
				useModelName := pm.config.Models[realModelName].UseModelName

				if useModelName != "" {
					fieldValue = useModelName
				} else {
					fieldValue = requestedModel
				}
			}
			field, err := multipartWriter.CreateFormField(key)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error recreating form field")
				return
			}
			if _, err = field.Write([]byte(fieldValue)); err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error writing form field")
				return
			}
		}
	}

	// Copy all files from the original request
	for key, fileHeaders := range c.Request.MultipartForm.File {
		for _, fileHeader := range fileHeaders {
			formFile, err := multipartWriter.CreateFormFile(key, fileHeader.Filename)
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error recreating form file")
				return
			}

			file, err := fileHeader.Open()
			if err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error opening uploaded file")
				return
			}

			if _, err = io.Copy(formFile, file); err != nil {
				file.Close()
				pm.sendErrorResponse(c, http.StatusInternalServerError, "error copying file data")
				return
			}
			file.Close()
		}
	}

	// Close the multipart writer to finalize the form
	if err := multipartWriter.Close(); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "error finalizing multipart form")
		return
	}

	// Create a new request with the reconstructed form data
	modifiedReq, err := http.NewRequestWithContext(
		c.Request.Context(),
		c.Request.Method,
		c.Request.URL.String(),
		&requestBuffer,
	)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "error creating modified request")
		return
	}

	// Copy the headers from the original request
	modifiedReq.Header = c.Request.Header.Clone()
	modifiedReq.Header.Set("Content-Type", multipartWriter.FormDataContentType())

	// set the content length of the body
	modifiedReq.Header.Set("Content-Length", strconv.Itoa(requestBuffer.Len()))
	modifiedReq.ContentLength = int64(requestBuffer.Len())

	// Use the modified request for proxying
	if err := processGroup.ProxyRequest(realModelName, c.Writer, modifiedReq); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying request: %s", err.Error()))
		pm.proxyLogger.Errorf("Error Proxying Request for processGroup %s and model %s", processGroup.id, realModelName)
		return
	}
}

func (pm *ProxyManager) sendErrorResponse(c *gin.Context, statusCode int, message string) {
	acceptHeader := c.GetHeader("Accept")

	if strings.Contains(acceptHeader, "application/json") {
		c.JSON(statusCode, gin.H{"error": message})
	} else {
		c.String(statusCode, message)
	}
}

func (pm *ProxyManager) unloadAllModelsHandler(c *gin.Context) {
	pm.StopProcesses(StopImmediately)
	c.String(http.StatusOK, "OK")
}

func (pm *ProxyManager) listRunningProcessesHandler(context *gin.Context) {
	context.Header("Content-Type", "application/json")
	runningProcesses := make([]gin.H, 0) // Default to an empty response.

	for _, processGroup := range pm.processGroups {
		for _, process := range processGroup.processes {
			if process.CurrentState() == StateReady {
				runningProcesses = append(runningProcesses, gin.H{
					"model": process.ID,
					"state": process.state,
				})
			}
		}
	}

	// Put the results under the `running` key.
	response := gin.H{
		"running": runningProcesses,
	}

	context.JSON(http.StatusOK, response) // Always return 200 OK
}

func (pm *ProxyManager) findGroupByModelName(modelName string) *ProcessGroup {
	for _, group := range pm.processGroups {
		if group.HasMember(modelName) {
			return group
		}
	}
	return nil
}

func (pm *ProxyManager) proxyOllamaPassthrough(c *gin.Context) {
	// Handle special endpoints that don't need model routing
	switch c.Request.URL.Path {
	case "/api/tags":
		pm.handleOllamaTagsRequest(c)
		return
	case "/api/ps":
		pm.handleOllamaPsRequest(c)
		return
	case "/api/show":
		pm.handleOllamaShowRequest(c)
		return
	}

	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "could not read request body")
		return
	}

	// Extract model name for routing
	var requestedModel string
	if len(bodyBytes) > 0 {
		requestedModel = gjson.GetBytes(bodyBytes, "model").String()
	}

	if requestedModel == "" {
		// No model specified - might be a list operation, pass through to Ollama
		pm.handleOllamaPassthroughRequest(c, bodyBytes, "")
		return
	}

	// Map the model name if needed
	realModelName, found := pm.config.RealModelName(requestedModel)
	if found {
		requestedModel = realModelName
	}

	// Check if this is an OAI model that needs translation
	if pm.ollamaTranslator.isOAIModel(requestedModel) {
		pm.handleOllamaToOAITranslation(c, bodyBytes, requestedModel)
		return
	}

	// It's a native Ollama model, use existing passthrough logic
	pm.handleOllamaPassthroughRequest(c, bodyBytes, requestedModel)
}

func (pm *ProxyManager) findOllamaGroup() *ProcessGroup {
	// Look for a group configured as Ollama backend
	for _, group := range pm.processGroups {
		if group.backendType == "ollama" {
			return group
		}
	}
	return nil
}

// New method to handle translation from Ollama to OAI

// Modified handler for /api/tags to include all models
func (pm *ProxyManager) handleOllamaTagsRequest(c *gin.Context) {
	models := []map[string]interface{}{}
	ollamaModelNames := make(map[string]bool) // Track Ollama model names to avoid duplicates

	// First, get native Ollama models if available
	ollamaGroup := pm.findOllamaGroup()
	if ollamaGroup != nil && ollamaGroup.baseURL != "" {
		// Fetch from real Ollama
		client := &http.Client{Timeout: 10 * time.Second}
		resp, err := client.Get(ollamaGroup.baseURL + "/api/tags")
		if err == nil {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				bodyBytes, _ := io.ReadAll(resp.Body)
				if ollamaModels := gjson.GetBytes(bodyBytes, "models"); ollamaModels.Exists() {
					ollamaModels.ForEach(func(_, value gjson.Result) bool {
						modelName := value.Get("name").String()
						ollamaModelNames[modelName] = true // Track this name

						model := map[string]interface{}{
							"name":        modelName,
							"model":       value.Get("model").String(),
							"modified_at": value.Get("modified_at").String(),
							"size":        value.Get("size").Int(),
							"digest":      value.Get("digest").String(),
							"details":     value.Get("details").Value(),
						}
						models = append(models, model)
						return true
					})
				}
			}
		}
	}

	// Add OAI models
	for modelID, modelConfig := range pm.config.Models {
		if modelConfig.Unlisted {
			continue
		}

		// Check if this is an OAI model (not in an Ollama group)
		group := pm.findGroupByModelName(modelID)
		if group != nil && group.backendType != "ollama" {
			// Skip if this name already exists as an Ollama model
			if ollamaModelNames[modelID] {
				continue
			}

			// Format as Ollama model entry - matching exact structure
			model := map[string]interface{}{
				"name":        modelID,
				"model":       modelID, // Add this field
				"modified_at": time.Now().Format(time.RFC3339),
				"size":        0,                   // Unknown for OAI models
				"digest":      "openai:" + modelID, // Fake digest to indicate it's an OAI model
				"details": map[string]interface{}{
					"parent_model":       "",
					"format":             "openai",
					"family":             "openai",
					"families":           []string{"openai"}, // Array format
					"parameter_size":     "unknown",
					"quantization_level": "none",
				},
			}

			// Add aliases as separate entries
			for _, alias := range modelConfig.Aliases {
				// Skip if alias conflicts with Ollama model name
				if ollamaModelNames[alias] {
					continue
				}

				aliasModel := map[string]interface{}{
					"name":        alias,
					"model":       alias, // Add this field
					"modified_at": time.Now().Format(time.RFC3339),
					"size":        0,
					"digest":      "openai:" + modelID, // Points to real model
					"details": map[string]interface{}{
						"parent_model":       "",
						"format":             "openai",
						"family":             "openai",
						"families":           []string{"openai"}, // Array format
						"parameter_size":     "unknown",
						"quantization_level": "none",
					},
				}
				models = append(models, aliasModel)
			}

			models = append(models, model)
		}
	}

	// Return combined list
	c.JSON(http.StatusOK, gin.H{
		"models": models,
	})
}

// Modified handler for /api/ps to show running models from all sources
func (pm *ProxyManager) handleOllamaPsRequest(c *gin.Context) {
	runningModels := []map[string]interface{}{}

	// Get Ollama running models if available
	ollamaGroup := pm.findOllamaGroup()
	if ollamaGroup != nil && ollamaGroup.baseURL != "" {
		client := &http.Client{Timeout: 10 * time.Second}
		resp, err := client.Get(ollamaGroup.baseURL + "/api/ps")
		if err == nil {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				bodyBytes, _ := io.ReadAll(resp.Body)
				if models := gjson.GetBytes(bodyBytes, "models"); models.Exists() {
					models.ForEach(func(_, value gjson.Result) bool {
						model := map[string]interface{}{
							"name":       value.Get("name").String(),
							"model":      value.Get("model").String(),
							"size":       value.Get("size").Int(),
							"digest":     value.Get("digest").String(),
							"expires_at": value.Get("expires_at").String(),
							"size_vram":  value.Get("size_vram").Int(),
						}
						runningModels = append(runningModels, model)
						return true
					})
				}
			}
		}
	}

	// Add running OAI models
	for _, processGroup := range pm.processGroups {
		if processGroup.backendType != "ollama" {
			for modelID, process := range processGroup.processes {
				if process.CurrentState() == StateReady {
					model := map[string]interface{}{
						"name":       modelID,
						"model":      modelID,
						"size":       0, // Unknown for OAI
						"digest":     "openai:" + modelID,
						"expires_at": time.Now().Add(time.Hour).Format(time.RFC3339), // Fake expiry
						"size_vram":  0,                                              // Unknown
					}
					runningModels = append(runningModels, model)
				}
			}
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"models": runningModels,
	})
}

// Modified handler for /api/show to handle OAI models
func (pm *ProxyManager) handleOllamaShowRequest(c *gin.Context) {
	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "could not read request body")
		return
	}

	requestedModel := gjson.GetBytes(bodyBytes, "name").String()
	if requestedModel == "" {
		requestedModel = gjson.GetBytes(bodyBytes, "model").String()
	}

	if requestedModel == "" {
		pm.sendErrorResponse(c, http.StatusBadRequest, "model name required")
		return
	}

	// Check if it's an OAI model
	realModelName, found := pm.config.RealModelName(requestedModel)
	if found {
		group := pm.findGroupByModelName(realModelName)
		if group != nil && group.backendType != "ollama" {
			// Return info about OAI model
			modelConfig := pm.config.Models[realModelName]

			response := gin.H{
				"modelfile":  fmt.Sprintf("# OpenAI Compatible Model: %s\n# Proxied through llama-swap", realModelName),
				"parameters": "temperature 0.7\ntop_p 0.9\ntop_k 40",
				"template":   "{{ .Prompt }}",
				"details": gin.H{
					"format":             "openai",
					"family":             "openai",
					"parameter_size":     "unknown",
					"quantization_level": "none",
				},
			}

			if modelConfig.Description != "" {
				response["description"] = modelConfig.Description
			}

			c.JSON(http.StatusOK, response)
			return
		}
	}

	// Otherwise, pass through to Ollama
	pm.handleOllamaPassthroughRequest(c, bodyBytes, requestedModel)
}

// New method to handle translation from Ollama to OAI
func (pm *ProxyManager) handleOllamaToOAITranslation(c *gin.Context, bodyBytes []byte, modelName string) {
	endpoint := c.Request.URL.Path

	// Check for unload request first
	keepAlive := gjson.GetBytes(bodyBytes, "keep_alive")
	if keepAlive.Exists() && keepAlive.Int() == 0 {
		prompt := gjson.GetBytes(bodyBytes, "prompt").String()
		messages := gjson.GetBytes(bodyBytes, "messages").Array()

		if prompt == "" && len(messages) == 0 {
			pm.ollamaTranslator.handleUnloadRequest(c, modelName)
			return
		}
	}

	// Translate the request based on endpoint
	var oaiReqBytes []byte
	var err error
	var isChat bool

	switch endpoint {
	case "/api/generate":
		oaiReqBytes, err = pm.ollamaTranslator.translateOllamaGenerateToOAI(bodyBytes)
		isChat = false
	case "/api/chat":
		oaiReqBytes, err = pm.ollamaTranslator.translateOllamaChatToOAI(bodyBytes)
		isChat = true
	default:
		pm.sendErrorResponse(c, http.StatusBadRequest, "unsupported Ollama endpoint for OAI model")
		return
	}

	if err != nil {
		if err.Error() == "unload_request" {
			pm.ollamaTranslator.handleUnloadRequest(c, modelName)
			return
		}
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("translation error: %s", err.Error()))
		return
	}

	// Create a response interceptor
	interceptor := &OllamaResponseInterceptor{
		ResponseWriter: c.Writer,
		translator:     pm.ollamaTranslator,
		originalModel:  modelName,
		isStreaming:    gjson.GetBytes(oaiReqBytes, "stream").Bool(),
		isChat:         isChat, // Set this field
	}

	// Create new request with translated body
	newReq, err := http.NewRequestWithContext(
		c.Request.Context(),
		"POST",
		"/v1/chat/completions",
		bytes.NewReader(oaiReqBytes),
	)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "failed to create translated request")
		return
	}

	// Copy headers
	newReq.Header = c.Request.Header.Clone()
	newReq.Header.Set("Content-Type", "application/json")
	newReq.Header.Set("Content-Length", strconv.Itoa(len(oaiReqBytes)))

	// Process through OAI handler with intercepted response
	c.Writer = interceptor
	c.Request = newReq
	pm.proxyOAIHandler(c)

	// Finalize the response translation
	interceptor.finalize()
}

// Response interceptor to translate OAI responses back to Ollama format
type OllamaResponseInterceptor struct {
	gin.ResponseWriter
	translator    *OllamaToOAITranslator
	originalModel string
	isStreaming   bool
	isChat        bool
	buffer        []byte
	statusCode    int
	headerWritten bool
}

func (i *OllamaResponseInterceptor) Write(data []byte) (int, error) {
	// Buffer the response for translation
	i.buffer = append(i.buffer, data...)
	return len(data), nil
}

func (i *OllamaResponseInterceptor) Flush() {
	// For streaming responses, translate and flush chunks
	if i.isStreaming && len(i.buffer) > 0 {
		translated, err := i.translator.translateOAIResponseToOllama(i.buffer, true, i.originalModel, i.isChat)
		if err == nil {
			i.ResponseWriter.Write(translated)
			if flusher, ok := i.ResponseWriter.(http.Flusher); ok {
				flusher.Flush()
			}
		}
		i.buffer = nil
	}
}

func (i *OllamaResponseInterceptor) CloseNotify() <-chan bool {
	if cn, ok := i.ResponseWriter.(http.CloseNotifier); ok {
		return cn.CloseNotify()
	}
	return nil
}

// Add this to handle the final response
func (i *OllamaResponseInterceptor) finalize() {
	if i.statusCode == 0 {
		i.statusCode = http.StatusOK
	}

	// Make sure headers are written
	if !i.headerWritten {
		i.WriteHeader(i.statusCode)
	}

	// Handle error responses
	if i.statusCode != http.StatusOK {
		i.ResponseWriter.Write(i.buffer)
		return
	}

	// Translate the response
	translated, err := i.translator.translateOAIResponseToOllama(i.buffer, i.isStreaming, i.originalModel, i.isChat)
	if err != nil {
		// Fall back to original response
		i.ResponseWriter.Write(i.buffer)
		return
	}

	// Write translated response
	i.ResponseWriter.Write(translated)
}

// Existing passthrough logic (refactored from original)
func (pm *ProxyManager) handleOllamaPassthroughRequest(c *gin.Context, bodyBytes []byte, requestedModel string) {
	// Find Ollama process group
	ollamaGroup := pm.findOllamaGroup()
	if ollamaGroup == nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, "no Ollama backend configured")
		return
	}

	// If we have a model, update the body with the real name
	if requestedModel != "" {
		bodyBytes, _ = sjson.SetBytes(bodyBytes, "model", requestedModel)

		// Send unload signal to other process groups
		pm.signalUnloadOtherGroups(ollamaGroup, requestedModel)
	}

	// Rebuild request body
	c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
	c.Request.Header.Del("transfer-encoding")
	c.Request.Header.Set("content-length", strconv.Itoa(len(bodyBytes)))
	c.Request.ContentLength = int64(len(bodyBytes))

	// Direct HTTP call to Ollama
	if err := pm.proxyDirectToOllama(ollamaGroup, c.Writer, c.Request); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error proxying to Ollama: %s", err.Error()))
		pm.proxyLogger.Errorf("Error Proxying Ollama Request for model %s: %v", requestedModel, err)
		return
	}
}

func (pm *ProxyManager) signalUnloadOtherGroups(activeGroup *ProcessGroup, newModel string) {
	// Only unload if the new group is exclusive
	if !activeGroup.exclusive {
		return
	}

	pm.proxyLogger.Debugf("Unloading other groups before loading Ollama model: %s", newModel)

	for groupId, otherGroup := range pm.processGroups {
		if groupId != activeGroup.id && !otherGroup.persistent {
			// For non-Ollama groups: stop processes
			if otherGroup.backendType != "ollama" {
				pm.proxyLogger.Debugf("Stopping non-Ollama group: %s", groupId)
				otherGroup.StopProcesses(StopWaitForInflightRequest)
			} else {
				// For other Ollama groups: send unload signal
				pm.proxyLogger.Debugf("Unloading other Ollama group: %s", groupId)
				go pm.sendOllamaUnloadAll(otherGroup)
			}
		}
	}
}
func (pm *ProxyManager) sendOllamaUnloadAll(ollamaGroup *ProcessGroup) {
	// Get running models from Ollama
	runningModels, err := pm.getOllamaRunningModels(ollamaGroup)
	if err != nil {
		pm.proxyLogger.Errorf("Failed to get running Ollama models: %v", err)
		return
	}

	// Send unload signal to each model
	for _, modelName := range runningModels {
		pm.proxyLogger.Debugf("Unloading Ollama model: %s", modelName)

		unloadRequest := map[string]interface{}{
			"model":      modelName,
			"messages":   []interface{}{},
			"keep_alive": 0,
		}

		// Make async call to unload
		go pm.callOllamaUnload(ollamaGroup, unloadRequest)
	}
}

func (pm *ProxyManager) getOllamaRunningModels(group *ProcessGroup) ([]string, error) {
	if group.baseURL == "" {
		return nil, fmt.Errorf("no base URL configured for group %s", group.id)
	}

	// Create HTTP client
	client := &http.Client{Timeout: 10 * time.Second}

	// Call /api/ps endpoint
	resp, err := client.Get(group.baseURL + "/api/ps")
	if err != nil {
		return nil, fmt.Errorf("failed to call /api/ps: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code from /api/ps: %d", resp.StatusCode)
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read /api/ps response: %v", err)
	}

	// Parse response to extract model names
	var models []string
	modelsArray := gjson.GetBytes(bodyBytes, "models")
	if modelsArray.Exists() {
		modelsArray.ForEach(func(_, value gjson.Result) bool {
			if modelName := value.Get("name").String(); modelName != "" {
				models = append(models, modelName)
			}
			return true
		})
	}

	return models, nil
}

func (pm *ProxyManager) callOllamaUnload(group *ProcessGroup, request map[string]interface{}) {
	if group.baseURL == "" {
		pm.proxyLogger.Errorf("No base URL configured for group %s", group.id)
		return
	}

	client := &http.Client{Timeout: 30 * time.Second}

	// Convert request to JSON
	requestBytes, err := json.Marshal(request)
	if err != nil {
		pm.proxyLogger.Errorf("Failed to marshal unload request: %v", err)
		return
	}

	// Make the unload request
	resp, err := client.Post(
		group.baseURL+"/api/chat",
		"application/json",
		bytes.NewBuffer(requestBytes),
	)
	if err != nil {
		pm.proxyLogger.Errorf("Failed to send unload request: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		pm.proxyLogger.Errorf("Unload request failed with status: %d", resp.StatusCode)
		return
	}

	// Log success
	modelName := request["model"].(string)
	pm.proxyLogger.Debugf("Successfully sent unload signal for model: %s", modelName)
}
func (pm *ProxyManager) proxyDirectToOllama(group *ProcessGroup, writer http.ResponseWriter, request *http.Request) error {
	if group.baseURL == "" {
		return fmt.Errorf("no base URL configured for Ollama group %s", group.id)
	}

	// Create the target URL
	targetURL := group.baseURL + request.URL.Path

	// Create new request to Ollama
	proxyReq, err := http.NewRequest(request.Method, targetURL, request.Body)
	if err != nil {
		return fmt.Errorf("failed to create proxy request: %v", err)
	}

	// Copy headers
	for name, values := range request.Header {
		for _, value := range values {
			proxyReq.Header.Add(name, value)
		}
	}

	// Make the request to Ollama
	client := &http.Client{Timeout: 300 * time.Second} // 5 minute timeout for long generations
	resp, err := client.Do(proxyReq)
	if err != nil {
		return fmt.Errorf("failed to proxy to Ollama: %v", err)
	}
	defer resp.Body.Close()

	// Copy response headers
	for name, values := range resp.Header {
		for _, value := range values {
			writer.Header().Add(name, value)
		}
	}

	// Set status code
	writer.WriteHeader(resp.StatusCode)

	// Copy response body
	_, err = io.Copy(writer, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to copy response body: %v", err)
	}

	return nil
}
