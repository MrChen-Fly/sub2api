package service

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// forwardResponsesAsChatCompletions converts a Responses API request to Chat
// Completions format, forwards to the upstream /v1/chat/completions endpoint,
// and converts the response back to Responses API format.
//
// This enables APIKey accounts whose upstream only supports /v1/chat/completions
// (DeepSeek, Kimi, GLM, etc.) to serve Responses API clients (Codex CLI, etc.).
func (s *OpenAIGatewayService) forwardResponsesAsChatCompletions(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	defaultMappedModel string,
) (*OpenAIForwardResult, error) {
	startTime := time.Now()

	// 1. Parse Responses request
	var responsesReq apicompat.ResponsesRequest
	if err := json.Unmarshal(body, &responsesReq); err != nil {
		writeResponsesError(c, http.StatusBadRequest, "invalid_request_error", "Failed to parse request body")
		return nil, fmt.Errorf("parse responses request: %w", err)
	}
	originalModel := responsesReq.Model
	clientStream := responsesReq.Stream

	// 2. Convert Responses → Chat Completions
	ccReq, err := apicompat.ResponsesRequestToChatCompletions(&responsesReq)
	if err != nil {
		writeResponsesError(c, http.StatusBadRequest, "invalid_request_error", "Failed to convert request: "+err.Error())
		return nil, fmt.Errorf("convert responses to chat completions: %w", err)
	}

	// 3. Debug: log original input items to diagnose missing reasoning_content
	s.logResponsesInputItemsForDebug(c, &responsesReq)

	// 4. Model mapping
	billingModel := resolveOpenAIForwardModel(account, originalModel, defaultMappedModel)
	upstreamModel := normalizeOpenAIModelForUpstream(account, billingModel)
	ccReq.Model = upstreamModel

	// 5. Marshal CC body
	ccBody, err := json.Marshal(ccReq)
	if err != nil {
		return nil, fmt.Errorf("marshal cc request: %w", err)
	}

	// 6. Apply fast policy
	updatedBody, policyErr := s.applyOpenAIFastPolicyToBody(ctx, account, upstreamModel, ccBody)
	if policyErr != nil {
		var blocked *OpenAIFastBlockedError
		if errors.As(policyErr, &blocked) {
			writeResponsesError(c, http.StatusForbidden, "permission_error", blocked.Message)
		}
		return nil, policyErr
	}
	ccBody = updatedBody

	// Force stream usage for billing
	if clientStream {
		ccBody, err = ensureOpenAIChatStreamUsage(ccBody)
		if err != nil {
			return nil, fmt.Errorf("enable stream usage: %w", err)
		}
	}

	logger.L().Info("openai responses_as_chat_completions: forwarding",
		zap.Int64("account_id", account.ID),
		zap.String("original_model", originalModel),
		zap.String("upstream_model", upstreamModel),
		zap.Bool("stream", clientStream),
		zap.Intp("max_tokens", ccReq.MaxTokens),
		zap.String("reasoning_effort", ccReq.ReasoningEffort),
	)

	// 7. Build upstream request
	apiKey := account.GetOpenAIApiKey()
	if apiKey == "" {
		return nil, fmt.Errorf("account %d missing api_key", account.ID)
	}
	baseURL := account.GetOpenAIBaseURL()
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}
	validatedURL, err := s.validateUpstreamBaseURL(baseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid base_url: %w", err)
	}
	targetURL := buildOpenAIChatCompletionsURL(validatedURL)

	upstreamCtx, releaseUpstreamCtx := detachUpstreamContext(ctx)
	upstreamReq, err := http.NewRequestWithContext(upstreamCtx, http.MethodPost, targetURL, bytes.NewReader(ccBody))
	releaseUpstreamCtx()
	if err != nil {
		return nil, fmt.Errorf("build upstream request: %w", err)
	}
	upstreamReq.Header.Set("Content-Type", "application/json")
	upstreamReq.Header.Set("Authorization", "Bearer "+apiKey)
	if clientStream {
		upstreamReq.Header.Set("Accept", "text/event-stream")
	} else {
		upstreamReq.Header.Set("Accept", "application/json")
	}
	customUA := account.GetOpenAIUserAgent()
	if customUA != "" {
		upstreamReq.Header.Set("user-agent", customUA)
	}

	// 8. Send request
	proxyURL := ""
	if account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	resp, err := s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		safeErr := sanitizeUpstreamErrorMessage(err.Error())
		setOpsUpstreamError(c, 0, safeErr, "")
		writeResponsesError(c, http.StatusBadGateway, "server_error", "Upstream request failed")
		return nil, fmt.Errorf("upstream request failed: %s", safeErr)
	}
	defer func() { _ = resp.Body.Close() }()

	// 9. Handle error response with failover
	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
		_ = resp.Body.Close()
		resp.Body = io.NopCloser(bytes.NewReader(respBody))

		upstreamMsg := strings.TrimSpace(extractUpstreamErrorMessage(respBody))
		upstreamMsg = sanitizeUpstreamErrorMessage(upstreamMsg)
		logger.L().Error("openai responses_as_chat: upstream error",
			zap.Int("status", resp.StatusCode),
			zap.String("message", upstreamMsg),
			zap.String("upstream_body", string(respBody)),
			zap.String("cc_body", string(ccBody)),
		)
		if s.shouldFailoverOpenAIUpstreamResponse(resp.StatusCode, upstreamMsg, respBody) {
			appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
				Platform:           account.Platform,
				AccountID:          account.ID,
				AccountName:        account.Name,
				UpstreamStatusCode: resp.StatusCode,
				UpstreamRequestID:  resp.Header.Get("x-request-id"),
				Kind:               "failover",
				Message:            upstreamMsg,
			})
			return nil, &UpstreamFailoverError{
				StatusCode:             resp.StatusCode,
				ResponseBody:           respBody,
				RetryableOnSameAccount: account.IsPoolMode() && (isPoolModeRetryableStatus(resp.StatusCode) || isOpenAITransientProcessingError(resp.StatusCode, upstreamMsg, respBody)),
			}
		}
		return handleResponsesAsChatErrorResponse(resp, c, account)
	}

	// 10. Forward response
	if clientStream {
		return s.streamResponsesFromChatCompletions(c, resp, originalModel, billingModel, upstreamModel, startTime)
	}
	return s.bufferResponsesFromChatCompletions(c, resp, originalModel, billingModel, upstreamModel, startTime)
}

// streamResponsesFromChatCompletions reads CC SSE chunks from upstream,
// converts to Responses SSE events, and writes them to the client.
func (s *OpenAIGatewayService) streamResponsesFromChatCompletions(
	c *gin.Context,
	resp *http.Response,
	originalModel string,
	billingModel string,
	upstreamModel string,
	startTime time.Time,
) (*OpenAIForwardResult, error) {
	requestID := resp.Header.Get("x-request-id")

	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	c.Writer.Header().Set("X-Reasoning-Included", "true")
	c.Writer.WriteHeader(http.StatusOK)

	scanner := bufio.NewScanner(resp.Body)
	maxLineSize := defaultMaxLineSize
	if s.cfg != nil && s.cfg.Gateway.MaxLineSize > 0 {
		maxLineSize = s.cfg.Gateway.MaxLineSize
	}
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineSize)

	state := apicompat.NewCCEventToResponsesState()
	var usage OpenAIUsage
	var firstTokenMs *int
	clientDisconnected := false

	for scanner.Scan() {
		line := scanner.Text()
		if payload, ok := extractOpenAISSEDataLine(line); ok {
			trimmed := strings.TrimSpace(payload)
			if trimmed == "[DONE]" {
				continue
			}

			var chunk apicompat.ChatCompletionsChunk
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				logger.L().Warn("openai responses_as_chat: failed to parse CC chunk",
					zap.Error(err),
					zap.String("request_id", requestID),
				)
				continue
			}

			// Track usage
			if chunk.Usage != nil {
				u := OpenAIUsage{
					InputTokens:  chunk.Usage.PromptTokens,
					OutputTokens: chunk.Usage.CompletionTokens,
				}
				if chunk.Usage.PromptTokensDetails != nil {
					u.CacheReadInputTokens = chunk.Usage.PromptTokensDetails.CachedTokens
				}
				usage = u
			}

			// Track first token
			if firstTokenMs == nil && chunkHasContent(&chunk) {
				elapsed := int(time.Since(startTime).Milliseconds())
				firstTokenMs = &elapsed
			}

			// Convert and emit
			events := apicompat.CCEventToResponsesEvents(&chunk, state)
			if !clientDisconnected {
				for _, evt := range events {
					sse, err := apicompat.CCChunkToResponsesSSE(evt)
					if err != nil {
						continue
					}
					if _, werr := c.Writer.WriteString(sse); werr != nil {
						clientDisconnected = true
						break
					}
				}
			}
			if len(events) > 0 && !clientDisconnected {
				c.Writer.Flush()
			}
		}
	}

	// Finalize: emit any pending close events
	finalEvents := apicompat.FinalizeCCToResponsesStream(state)
	if !clientDisconnected {
		for _, evt := range finalEvents {
			sse, err := apicompat.CCChunkToResponsesSSE(evt)
			if err != nil {
				continue
			}
			if _, werr := c.Writer.WriteString(sse); werr != nil {
				clientDisconnected = true
				break
			}
		}
	}
	if !clientDisconnected {
		c.Writer.Flush()
	}

	if err := scanner.Err(); err != nil {
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			logger.L().Warn("openai responses_as_chat: stream read error",
				zap.Error(err),
				zap.String("request_id", requestID),
			)
		}
	}

	return &OpenAIForwardResult{
		RequestID:        requestID,
		Usage:            usage,
		Model:            originalModel,
		BillingModel:     billingModel,
		UpstreamModel:    upstreamModel,
		Stream:           true,
		Duration:         time.Since(startTime),
		FirstTokenMs:     firstTokenMs,
		UpstreamEndpoint: "/v1/chat/completions",
	}, nil
}

// bufferResponsesFromChatCompletions reads the full CC JSON response and
// converts it to a Responses JSON response.
func (s *OpenAIGatewayService) bufferResponsesFromChatCompletions(
	c *gin.Context,
	resp *http.Response,
	originalModel string,
	billingModel string,
	upstreamModel string,
	startTime time.Time,
) (*OpenAIForwardResult, error) {
	requestID := resp.Header.Get("x-request-id")

	respBody, err := ReadUpstreamResponseBody(resp.Body, s.cfg, c, openAITooLargeError)
	if err != nil {
		if !errors.Is(err, ErrUpstreamResponseBodyTooLarge) {
			writeResponsesError(c, http.StatusBadGateway, "server_error", "Failed to read upstream response")
		}
		return nil, fmt.Errorf("read upstream body: %w", err)
	}

	var ccResp apicompat.ChatCompletionsResponse
	var usage OpenAIUsage
	if err := json.Unmarshal(respBody, &ccResp); err != nil {
		writeResponsesError(c, http.StatusBadGateway, "server_error", "Failed to parse upstream response")
		return nil, fmt.Errorf("parse cc response: %w", err)
	}
	if ccResp.Usage != nil {
		usage = OpenAIUsage{
			InputTokens:  ccResp.Usage.PromptTokens,
			OutputTokens: ccResp.Usage.CompletionTokens,
		}
		if ccResp.Usage.PromptTokensDetails != nil {
			usage.CacheReadInputTokens = ccResp.Usage.PromptTokensDetails.CachedTokens
		}
	}

	responsesResp := apicompat.ChatCompletionsToResponsesResponse(&ccResp)
	responsesResp.Model = originalModel // report original model to client

	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.Writer.Header().Set("Content-Type", "application/json")
	c.JSON(http.StatusOK, responsesResp)

	return &OpenAIForwardResult{
		RequestID:        requestID,
		Usage:            usage,
		Model:            originalModel,
		BillingModel:     billingModel,
		UpstreamModel:    upstreamModel,
		Stream:           false,
		Duration:         time.Since(startTime),
		UpstreamEndpoint: "/v1/chat/completions",
	}, nil
}

// handleResponsesAsChatErrorResponse writes an upstream error in Responses format.
func handleResponsesAsChatErrorResponse(resp *http.Response, c *gin.Context, account *Account) (*OpenAIForwardResult, error) {
	return nil, &UpstreamFailoverError{
		StatusCode:   resp.StatusCode,
		ResponseBody: nil,
	}
}

// chunkHasContent returns true if the CC chunk contains actual content (not
// just a role chunk or empty content).
func chunkHasContent(chunk *apicompat.ChatCompletionsChunk) bool {
	if chunk == nil || len(chunk.Choices) == 0 {
		return false
	}
	delta := chunk.Choices[0].Delta
	if delta.Content != nil && *delta.Content != "" {
		return true
	}
	if len(delta.ToolCalls) > 0 {
		return true
	}
	return false
}

// logResponsesInputItemsForDebug logs a concise summary of each input item in
// the Responses request. This helps diagnose missing reasoning_content issues
// by showing the exact Type/Role ordering and content sizes.
func (s *OpenAIGatewayService) logResponsesInputItemsForDebug(c *gin.Context, req *apicompat.ResponsesRequest) {
	var items []apicompat.ResponsesInputItem
	if err := json.Unmarshal(req.Input, &items); err != nil {
		// Input might be a plain string — skip verbose debug
		logger.L().Info("openai responses_as_chat: input is string, not items")
		return
	}
	logger.L().Info("openai responses_as_chat: input items",
		zap.Int("count", len(items)),
	)
	for i, item := range items {
		contentLen := len(item.Content)
		summaryLen := 0
		for _, s := range item.Summary {
			summaryLen += len(s.Text)
		}
		logger.L().Info("openai responses_as_chat: input item",
			zap.Int("idx", i),
			zap.String("type", item.Type),
			zap.String("role", item.Role),
			zap.Int("content_len", contentLen),
			zap.Int("summary_len", summaryLen),
			zap.Int("encrypted_len", len(item.EncryptedContent)),
			zap.String("call_id", item.CallID),
			zap.String("name", item.Name),
			zap.Int("output_len", len(item.Output)),
			zap.String("id", item.ID),
		)
	}
}
