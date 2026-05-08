package apicompat

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"go.uber.org/zap"
)

// ResponsesRequestToChatCompletions converts an OpenAI Responses API request
// into an OpenAI Chat Completions request. This enables APIKey accounts whose
// upstream only supports /v1/chat/completions (DeepSeek, Kimi, GLM, etc.) to
// serve Responses API clients (Codex CLI, etc.) by translating the protocol.
func ResponsesRequestToChatCompletions(req *ResponsesRequest) (*ChatCompletionsRequest, error) {
	messages, err := convertResponsesInputToChatMessages(req.Input)
	if err != nil {
		return nil, fmt.Errorf("convert input to messages: %w", err)
	}

	// Prepend instructions as a system message
	if strings.TrimSpace(req.Instructions) != "" {
		messages = append([]ChatMessage{{
			Role:    "system",
			Content: marshalStringAsContent(req.Instructions),
		}}, messages...)
	}

	out := &ChatCompletionsRequest{
		Model:           req.Model,
		Messages:        messages,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		Stream:          req.Stream,
		ServiceTier:     req.ServiceTier,
		ReasoningEffort: "",
	}

	if req.MaxOutputTokens != nil && *req.MaxOutputTokens > 0 {
		v := *req.MaxOutputTokens
		out.MaxTokens = &v
	}
	if out.MaxTokens == nil {
		// The upstream /v1/models endpoint (DeepSeek, Kimi, etc.) doesn't
		// return model capabilities, so we can't query dynamically. But we
		// DO know the original model's max_output_tokens from OpenAI docs.
		// Use that as the default so Codex gets what it expects.
		if v := lookupOriginalModelMaxTokens(req.Model); v > 0 {
			out.MaxTokens = &v
		}
	}

	if req.Reasoning != nil && req.Reasoning.Effort != "" {
		out.ReasoningEffort = req.Reasoning.Effort
	}

	if len(req.Tools) > 0 {
		out.Tools = convertResponsesToolsToChat(req.Tools)
	}

	if len(req.ToolChoice) > 0 {
		out.ToolChoice = req.ToolChoice
	}

	return out, nil
}

func marshalStringAsContent(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return json.RawMessage(b)
}

// convertResponsesInputToChatMessages converts a Responses input array (JSON)
// into a slice of ChatMessage values.
func convertResponsesInputToChatMessages(inputRaw json.RawMessage) ([]ChatMessage, error) {
	var s string
	if err := json.Unmarshal(inputRaw, &s); err == nil {
		return []ChatMessage{{Role: "user", Content: marshalStringAsContent(s)}}, nil
	}

	var items []ResponsesInputItem
	if err := json.Unmarshal(inputRaw, &items); err != nil {
		return nil, fmt.Errorf("parse input items: %w", err)
	}

	var msgs []ChatMessage
	var pendingReasoning string  // accumulated reasoning for the next assistant message
	lastAssistantClosed := false // true after function_call_output — next reasoning goes to new assistant
	for _, item := range items {
		switch {
		case item.Type == "function_call":
			// Attach to the last assistant message, or create one.
			idx := findLastAssistantIndex(msgs)
			tc := ChatToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: ChatFunctionCall{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			}
			if idx >= 0 && !lastAssistantClosed {
				msgs[idx].ToolCalls = append(msgs[idx].ToolCalls, tc)
			} else {
				msg := ChatMessage{
					Role:      "assistant",
					Content:   marshalStringAsContent(""),
					ToolCalls: []ChatToolCall{tc},
				}
				if pendingReasoning != "" {
					msg.ReasoningContent = pendingReasoning
					pendingReasoning = ""
				}
				msgs = append(msgs, msg)
				lastAssistantClosed = false
			}

		case item.Type == "function_call_output",
			item.Type == "mcp_tool_call_output",
			item.Type == "custom_tool_call_output":
			output := toolOutputAsString(item.Output)
			msgs = append(msgs, ChatMessage{
				Role:       "tool",
				ToolCallID: item.CallID,
				Content:    marshalStringAsContent(output),
			})
			lastAssistantClosed = true

		case item.Type == "reasoning":
			// DeepSeek requires reasoning_content to be passed back
			// in multi-turn conversations. Attach to the last
			// assistant message, or hold for the next one.
			text := reasoningContentText(item)
			if text != "" {
				idx := findLastAssistantIndex(msgs)
				if idx >= 0 && !lastAssistantClosed {
					if msgs[idx].ReasoningContent != "" {
						msgs[idx].ReasoningContent += "\n" + text
					} else {
						msgs[idx].ReasoningContent = text
					}
				} else {
					if pendingReasoning != "" {
						pendingReasoning += "\n" + text
					} else {
						pendingReasoning = text
					}
				}
			}

		case item.Role == "developer":
			content := convertResponsesContentToChatContent(item.Content)
			msgs = append(msgs, ChatMessage{Role: "system", Content: content})

		case item.Role == "system":
			content := convertResponsesContentToChatContent(item.Content)
			msgs = append(msgs, ChatMessage{Role: "system", Content: content})

		case item.Role == "user":
			content := convertResponsesContentToChatContent(item.Content)
			msgs = append(msgs, ChatMessage{Role: "user", Content: content})

		case item.Role == "assistant":
			content, embeddedRC := convertAssistantContentWithReasoning(item.Content)
			msg := ChatMessage{Role: "assistant", Content: content, ReasoningContent: embeddedRC}
			if pendingReasoning != "" {
				if msg.ReasoningContent != "" {
					msg.ReasoningContent = pendingReasoning + "\n" + msg.ReasoningContent
				} else {
					msg.ReasoningContent = pendingReasoning
				}
				pendingReasoning = ""
			}
			msgs = append(msgs, msg)
			lastAssistantClosed = false

		default:
			// Unknown item type/role — skip with a warning so operators can catch protocol drift.
			logger.L().Warn("responses_as_chat: skipping unknown input item",
				zap.String("type", item.Type),
				zap.String("role", item.Role),
			)
		}
	}

	// Post-process: ensure reasoning_content on assistant messages with tool_calls.
	// DeepSeek requires reasoning_content for multi-turn tool-calling conversations.
	// Codex CLI strips reasoning_text content parts from assistant message content,
	// so the embedded-reasoning approach is unreliable. Fall back to the message
	// text, or synthesize from tool call names when content is empty.
	for i := range msgs {
		if msgs[i].Role == "assistant" && len(msgs[i].ToolCalls) > 0 && msgs[i].ReasoningContent == "" {
			rc := extractContentText(msgs[i].Content)
			if rc == "" {
				names := make([]string, len(msgs[i].ToolCalls))
				for j, tc := range msgs[i].ToolCalls {
					names[j] = tc.Function.Name
				}
				rc = "Calling: " + strings.Join(names, ", ")
			}
			msgs[i].ReasoningContent = rc
		}
	}

	msgs = reorderToolMessages(msgs)

	return msgs, nil
}

// reorderToolMessages moves tool messages to immediately follow their
// owning assistant message. In Responses API input, function_call_output
// items may be separated from their function_calls by intervening messages
// (e.g. a user reply). Chat Completions requires tool messages to directly
// follow the assistant that owns the tool_call_id.
func reorderToolMessages(msgs []ChatMessage) []ChatMessage {
	ownerIdx := make(map[string]bool)
	for _, m := range msgs {
		if m.Role == "assistant" {
			for _, tc := range m.ToolCalls {
				ownerIdx[tc.ID] = true
			}
		}
	}
	if len(ownerIdx) == 0 {
		return msgs
	}

	// Separate tool messages; insert each after its owning assistant.
	var out []ChatMessage
	var tools []ChatMessage
	for _, m := range msgs {
		if m.Role == "tool" {
			tools = append(tools, m)
		} else {
			out = append(out, m)
		}
	}
	// Process in reverse so multiple tools for the same assistant
	// preserve their original order after sequential insertions.
	for i := len(tools) - 1; i >= 0; i-- {
		tm := tools[i]
		_, ok := ownerIdx[tm.ToolCallID]
		if !ok {
			out = append([]ChatMessage{tm}, out...)
			continue
		}
		// Find where the owner assistant ended up in out.
		insertAt := -1
		for j, m := range out {
			if m.Role == "assistant" {
				for _, tc := range m.ToolCalls {
					if tc.ID == tm.ToolCallID {
						insertAt = j + 1
						break
					}
				}
			}
			if insertAt >= 0 {
				break
			}
		}
		if insertAt >= 0 {
			out = append(out[:insertAt], append([]ChatMessage{tm}, out[insertAt:]...)...)
		} else {
			out = append(out, tm)
		}
	}
	return out
}

// findLastAssistantIndex returns the index of the last assistant message in
// the slice, or -1 if none exists.
func findLastAssistantIndex(msgs []ChatMessage) int {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "assistant" {
			return i
		}
	}
	return -1
}

// reasoningContentText extracts reasoning text from all available fields.
func reasoningContentText(item ResponsesInputItem) string {
	// 1. Summary field (primary source for reasoning display text)
	if len(item.Summary) > 0 {
		var texts []string
		for _, sum := range item.Summary {
			if sum.Text != "" {
				texts = append(texts, sum.Text)
			}
		}
		if len(texts) > 0 {
			return strings.Join(texts, "\n")
		}
	}
	// 2. Content field (full reasoning content with reasoning_text parts)
	text := extractTextFromContent(item.Content)
	if text != "" {
		return text
	}
	// 3. EncryptedContent as fallback
	if item.EncryptedContent != "" {
		return item.EncryptedContent
	}
	return ""
}

// convertResponsesContentToChatContent converts a Responses content field
// (string or []ResponsesContentPart) to Chat content format.
func convertResponsesContentToChatContent(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return marshalStringAsContent("")
	}

	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return raw // already a string
	}

	var parts []ResponsesContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return raw // unknown format, pass through
	}

	if len(parts) == 0 {
		return marshalStringAsContent("")
	}

	// If only text parts, collapse to a plain string for simplicity
	if allTextParts(parts) {
		var b strings.Builder
		for _, p := range parts {
			b.WriteString(p.Text)
		}
		return marshalStringAsContent(b.String())
	}

	// Multi-modal: convert to ChatContentPart array
	var chatParts []ChatContentPart
	for _, p := range parts {
		switch p.Type {
		case "input_text", "output_text", "text":
			chatParts = append(chatParts, ChatContentPart{Type: "text", Text: p.Text})
		case "input_image":
			chatParts = append(chatParts, ChatContentPart{
				Type: "image_url",
				ImageURL: &ChatImageURL{
					URL:    p.ImageURL,
					Detail: "auto",
				},
			})
		}
	}

	b, _ := json.Marshal(chatParts)
	return json.RawMessage(b)
}

func allTextParts(parts []ResponsesContentPart) bool {
	for _, p := range parts {
		if p.Type == "reasoning_text" {
			continue // embedded reasoning, not display text
		}
		if p.Type != "input_text" && p.Type != "output_text" && p.Type != "text" {
			return false
		}
	}
	return true
}

// convertAssistantContentWithReasoning extracts both display content and
// embedded reasoning_text from an assistant message's content field.
// reasoning_text content parts are extracted as reasoning_content for CC
// while output_text parts become the message content.
func convertAssistantContentWithReasoning(raw json.RawMessage) (json.RawMessage, string) {
	if len(raw) == 0 {
		return marshalStringAsContent(""), ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return raw, "" // plain string, no embedded reasoning
	}
	var parts []ResponsesContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return raw, "" // unknown format, pass through
	}
	var textParts []ResponsesContentPart
	var reasoningParts []string
	for _, p := range parts {
		switch p.Type {
		case "reasoning_text":
			if strings.TrimSpace(p.Text) != "" {
				reasoningParts = append(reasoningParts, p.Text)
			}
		default:
			textParts = append(textParts, p)
		}
	}
	reasoning := strings.Join(reasoningParts, "\n")
	if len(textParts) == 0 {
		return marshalStringAsContent(""), reasoning
	}
	if allTextParts(textParts) && len(textParts) == 1 && textParts[0].Type != "input_image" {
		return marshalStringAsContent(textParts[0].Text), reasoning
	}
	// Multi-part: marshal textParts as CC content array
	var chatParts []ChatContentPart
	for _, p := range textParts {
		switch p.Type {
		case "input_text", "output_text", "text":
			chatParts = append(chatParts, ChatContentPart{Type: "text", Text: p.Text})
		case "input_image":
			chatParts = append(chatParts, ChatContentPart{
				Type: "image_url",
				ImageURL: &ChatImageURL{URL: p.ImageURL, Detail: "auto"},
			})
		}
	}
	b, _ := json.Marshal(chatParts)
	return json.RawMessage(b), reasoning
}

// convertResponsesToolsToChat maps Responses API tools to Chat Completions
// tools. Non-function tools (namespace, etc.) are dropped. web_search is
// converted to a function tool so models (DeepSeek etc.) can call it.
func convertResponsesToolsToChat(tools []ResponsesTool) []ChatTool {
	var out []ChatTool
	for _, t := range tools {
		switch t.Type {
		case "function":
			out = append(out, ChatTool{
				Type: "function",
				Function: &ChatFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.Parameters,
					Strict:      t.Strict,
				},
			})
		case "web_search", "google_search", "web_search_20250305":
			out = append(out, ChatTool{
				Type: "function",
				Function: &ChatFunction{
					Name:        "web_search",
					Description: "Search the web for current information. Returns search result snippets.",
					Parameters:  webSearchParameters(),
				},
			})
		case "namespace":
			// Convert MCP namespace to a single function tool so Chat-only
			// models (DeepSeek etc.) can call MCP operations through it.
			// The response converter sets namespace on mcp__-prefixed calls.
			if t.Name != "" {
				out = append(out, ChatTool{
					Type: "function",
					Function: &ChatFunction{
						Name:        t.Name,
						Description: "MCP server: " + t.Name + ". Call with tool name and arguments.",
						Parameters:  mcpNamespaceSchema(),
					},
				})
			}
		}
		// other types are dropped
	}
	return out
}

// webSearchParameters returns the JSON Schema for the web_search function tool.
func webSearchParameters() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]}`)
}


// toolOutputAsString converts a tool output (string, object, or array) to a
// string suitable for CC tool messages.
func toolOutputAsString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "(empty)"
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	return string(raw)
}

// modelMaxOutputTokens maps original model name prefixes to their maximum
// output tokens. Model names from Codex may include snapshot suffixes
// (e.g. gpt-5.5-2026-04-23), so we match by prefix.
//
// Values are from OpenAI's official model documentation.
var modelMaxOutputTokens = map[string]int{
	"gpt-5.5":       128000,
	"gpt-5.5-pro":   128000,
	"gpt-5.4":       128000,
	"gpt-5.3-codex": 16384,
	"gpt-5.1-codex": 16384,
	"gpt-5.4-mini":  16384,
	"gpt-5.4-nano":  16384,
}

func lookupOriginalModelMaxTokens(model string) int {
	for prefix, max := range modelMaxOutputTokens {
		if strings.HasPrefix(model, prefix) {
			return max
		}
	}
	return 0
}

func mcpNamespaceSchema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{},"additionalProperties":true}`)
}
