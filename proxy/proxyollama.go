package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

// OllamaToOAITranslator handles translation between Ollama and OpenAI APIs
type OllamaToOAITranslator struct {
	pm *ProxyManager
}

// isOAIModel checks if a model should be handled by OpenAI API
func (t *OllamaToOAITranslator) isOAIModel(modelName string) bool {
	// Check if it's in an Ollama group
	if group := t.pm.findGroupByModelName(modelName); group != nil {
		return group.backendType != "ollama"
	}
	// If we can't find the model, assume it's not OAI
	return false
}

// translateOllamaGenerateToOAI converts Ollama /api/generate request to OpenAI format
func (t *OllamaToOAITranslator) translateOllamaGenerateToOAI(ollamaReq []byte) ([]byte, error) {
	// Parse Ollama request
	model := gjson.GetBytes(ollamaReq, "model").String()
	prompt := gjson.GetBytes(ollamaReq, "prompt").String()
	stream := gjson.GetBytes(ollamaReq, "stream").Bool()

	// Check for unload request
	keepAlive := gjson.GetBytes(ollamaReq, "keep_alive")
	if keepAlive.Exists() && keepAlive.Int() == 0 && prompt == "" {
		// This is an unload request - we'll handle this separately
		return nil, fmt.Errorf("unload_request")
	}

	// Build OpenAI request
	oaiReq := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"stream": stream,
	}

	// Copy over temperature if present
	if temp := gjson.GetBytes(ollamaReq, "temperature"); temp.Exists() {
		oaiReq["temperature"] = temp.Float()
	}

	// Copy over max_tokens from num_predict
	if numPredict := gjson.GetBytes(ollamaReq, "options.num_predict"); numPredict.Exists() {
		oaiReq["max_tokens"] = numPredict.Int()
	}

	return json.Marshal(oaiReq)
}

// translateOllamaChatToOAI converts Ollama /api/chat request to OpenAI format
func (t *OllamaToOAITranslator) translateOllamaChatToOAI(ollamaReq []byte) ([]byte, error) {
	// Parse Ollama request
	model := gjson.GetBytes(ollamaReq, "model").String()
	messages := gjson.GetBytes(ollamaReq, "messages").Array()
	stream := gjson.GetBytes(ollamaReq, "stream").Bool()

	// Check for unload request
	keepAlive := gjson.GetBytes(ollamaReq, "keep_alive")
	if keepAlive.Exists() && keepAlive.Int() == 0 && len(messages) == 0 {
		return nil, fmt.Errorf("unload_request")
	}

	// Convert messages format
	oaiMessages := make([]map[string]string, 0, len(messages))
	for _, msg := range messages {
		oaiMsg := map[string]string{
			"role":    msg.Get("role").String(),
			"content": msg.Get("content").String(),
		}
		oaiMessages = append(oaiMessages, oaiMsg)
	}

	// Build OpenAI request
	oaiReq := map[string]interface{}{
		"model":    model,
		"messages": oaiMessages,
		"stream":   stream,
	}

	// Copy over optional parameters
	if temp := gjson.GetBytes(ollamaReq, "temperature"); temp.Exists() {
		oaiReq["temperature"] = temp.Float()
	}

	if maxTokens := gjson.GetBytes(ollamaReq, "options.num_predict"); maxTokens.Exists() {
		oaiReq["max_tokens"] = maxTokens.Int()
	}

	return json.Marshal(oaiReq)
}

// translateOAIResponseToOllama converts OpenAI response to Ollama format
func (t *OllamaToOAITranslator) translateOAIResponseToOllama(oaiResp []byte, isStreaming bool, originalModel string, isChat bool) ([]byte, error) {
	if isStreaming {
		return t.translateStreamingResponse(oaiResp, originalModel, isChat)
	}
	return t.translateNonStreamingResponse(oaiResp, originalModel, isChat)
}

// translateNonStreamingResponse handles non-streaming responses
func (t *OllamaToOAITranslator) translateNonStreamingResponse(oaiResp []byte, model string, isChat bool) ([]byte, error) {
	// Parse OpenAI response
	content := gjson.GetBytes(oaiResp, "choices.0.message.content").String()
	finishReason := gjson.GetBytes(oaiResp, "choices.0.finish_reason").String()
	role := gjson.GetBytes(oaiResp, "choices.0.message.role").String()

	var ollamaResp map[string]interface{}

	if isChat {
		// Chat format includes message object
		ollamaResp = map[string]interface{}{
			"model":      model,
			"created_at": time.Now().Format(time.RFC3339),
			"message": map[string]interface{}{
				"role":    role,
				"content": content,
			},
			"done": true,
		}
	} else {
		// Generate format has response as string
		ollamaResp = map[string]interface{}{
			"model":      model,
			"created_at": time.Now().Format(time.RFC3339),
			"response":   content,
			"done":       true,
		}
	}

	// Map finish reason
	switch finishReason {
	case "stop":
		ollamaResp["done_reason"] = "stop"
	case "length":
		ollamaResp["done_reason"] = "length"
	default:
		if finishReason != "" {
			ollamaResp["done_reason"] = finishReason
		}
	}

	// Add usage stats if available
	if usage := gjson.GetBytes(oaiResp, "usage"); usage.Exists() {
		promptTokens := usage.Get("prompt_tokens").Int()
		completionTokens := usage.Get("completion_tokens").Int()
		totalTokens := usage.Get("total_tokens").Int()

		ollamaResp["prompt_eval_count"] = promptTokens
		ollamaResp["eval_count"] = completionTokens
		ollamaResp["total_duration"] = totalTokens * 1000000       // Fake nanoseconds
		ollamaResp["load_duration"] = 1000000                      // 1ms fake
		ollamaResp["prompt_eval_duration"] = promptTokens * 500000 // Fake timing
		ollamaResp["eval_duration"] = completionTokens * 1000000   // Fake timing
	}

	return json.Marshal(ollamaResp)
}

// translateStreamingResponse handles streaming responses
func (t *OllamaToOAITranslator) translateStreamingResponse(oaiResp []byte, model string, isChat bool) ([]byte, error) {
	var result bytes.Buffer

	// Process each SSE line
	lines := bytes.Split(oaiResp, []byte("\n"))
	for _, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		// Skip non-data lines
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		data := bytes.TrimPrefix(line, []byte("data: "))

		// Handle [DONE] marker
		if bytes.Equal(data, []byte("[DONE]")) {
			// Send final Ollama response with metrics
			var finalResp map[string]interface{}

			if isChat {
				finalResp = map[string]interface{}{
					"model":      model,
					"created_at": time.Now().Format(time.RFC3339),
					"done":       true,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "",
					},
					"done_reason": "stop",
				}
			} else {
				finalResp = map[string]interface{}{
					"model":       model,
					"created_at":  time.Now().Format(time.RFC3339),
					"done":        true,
					"response":    "",
					"done_reason": "stop",
				}
			}

			// Add fake metrics
			finalResp["total_duration"] = 1000000000      // 1 second
			finalResp["load_duration"] = 50000000         // 50ms
			finalResp["prompt_eval_duration"] = 100000000 // 100ms
			finalResp["eval_duration"] = 850000000        // 850ms

			finalJSON, _ := json.Marshal(finalResp)
			result.Write(finalJSON)
			result.WriteByte('\n')
			continue
		}

		// Parse OpenAI chunk
		if !gjson.ValidBytes(data) {
			continue
		}

		parsed := gjson.ParseBytes(data)
		content := parsed.Get("choices.0.delta.content").String()
		role := parsed.Get("choices.0.delta.role").String()
		finishReason := parsed.Get("choices.0.finish_reason").String()

		// Skip empty content chunks unless it's the final chunk
		if content == "" && finishReason == "" && role == "" {
			continue
		}

		// Build Ollama chunk
		var ollamaChunk map[string]interface{}

		if isChat {
			message := map[string]interface{}{}
			if role != "" {
				message["role"] = role
			} else {
				message["role"] = "assistant"
			}
			message["content"] = content

			ollamaChunk = map[string]interface{}{
				"model":      model,
				"created_at": time.Now().Format(time.RFC3339),
				"message":    message,
				"done":       false,
			}
		} else {
			ollamaChunk = map[string]interface{}{
				"model":      model,
				"created_at": time.Now().Format(time.RFC3339),
				"response":   content,
				"done":       false,
			}
		}

		chunkJSON, _ := json.Marshal(ollamaChunk)
		result.Write(chunkJSON)
		result.WriteByte('\n')
	}

	return result.Bytes(), nil
}

// handleUnloadRequest processes model unload requests
func (t *OllamaToOAITranslator) handleUnloadRequest(c *gin.Context, modelName string) {
	// Find the process group
	processGroup := t.pm.findGroupByModelName(modelName)
	if processGroup == nil {
		t.pm.sendErrorResponse(c, http.StatusNotFound, fmt.Sprintf("model %s not found", modelName))
		return
	}

	// Stop the process
	if process, exists := processGroup.processes[modelName]; exists {
		process.StopImmediately()
	}

	// Send Ollama unload response
	response := map[string]interface{}{
		"model":       modelName,
		"created_at":  time.Now().Format(time.RFC3339),
		"response":    "",
		"done":        true,
		"done_reason": "unload",
	}

	c.JSON(http.StatusOK, response)
}
