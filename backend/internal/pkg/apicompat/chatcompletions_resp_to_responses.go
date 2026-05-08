package apicompat

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Non-streaming: ChatCompletionsResponse → ResponsesResponse
// ---------------------------------------------------------------------------

// ChatCompletionsToResponsesResponse converts a Chat Completions response into
// a Responses API response.
func ChatCompletionsToResponsesResponse(resp *ChatCompletionsResponse) *ResponsesResponse {
	id := resp.ID
	if id == "" {
		id = generateResponsesID()
	}

	out := &ResponsesResponse{
		ID:     id,
		Object: "response",
		Model:  resp.Model,
	}

	var outputs []ResponsesOutput

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]

		// Reasoning is emitted as a separate output item, not embedded
		// in message content (reasoning_text is output-only and causes
		// "unknown variant" errors when it round-trips into input).
		outputText := extractContentText(choice.Message.Content)
		reasoningText := choice.Message.ReasoningContent
		if strings.TrimSpace(reasoningText) != "" {
			outputs = append(outputs, ResponsesOutput{
				Type:             "reasoning",
				ID:               generateItemID(),
				EncryptedContent: base64.StdEncoding.EncodeToString([]byte(reasoningText)),
				Content:          []ResponsesContentPart{{Type: "reasoning_text", Text: reasoningText}},
				Summary:          []ResponsesSummary{{Type: "summary_text", Text: reasoningText}},
				Status:           "completed",
			})
		}
		msgParts := buildMessageContentParts("", outputText)
		outputs = append(outputs, ResponsesOutput{
			Type:    "message",
			ID:      generateItemID(),
			Role:    "assistant",
			Content: msgParts,
			Status:  "completed",
		})

		// Add function_call outputs for tool calls
		for _, tc := range choice.Message.ToolCalls {
			args := tc.Function.Arguments
			if args == "" {
				args = "{}"
			}
			outputType := "function_call"
			if tc.Function.Name == "web_search" {
				outputType = "web_search_call"
			}
			outputs = append(outputs, ResponsesOutput{
				Type:      outputType,
				ID:        generateItemID(),
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Namespace: extractMCPNamespace(tc.Function.Name),
				Arguments: args,
				Status:    "completed",
			})
		}
	}

	if len(outputs) == 0 {
		outputs = append(outputs, ResponsesOutput{
			Type:    "message",
			ID:      generateItemID(),
			Role:    "assistant",
			Content: []ResponsesContentPart{{Type: "output_text", Text: ""}},
			Status:  "completed",
		})
	}
	out.Output = outputs

	// Status mapping
	if len(resp.Choices) > 0 {
		switch resp.Choices[0].FinishReason {
		case "stop", "tool_calls":
			out.Status = "completed"
		case "length":
			out.Status = "incomplete"
			out.IncompleteDetails = &ResponsesIncompleteDetails{Reason: "max_output_tokens"}
		default:
			out.Status = "completed"
		}
	} else {
		out.Status = "completed"
	}

	// Usage
	if resp.Usage != nil {
		out.Usage = &ResponsesUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		}
		if resp.Usage.PromptTokensDetails != nil && resp.Usage.PromptTokensDetails.CachedTokens > 0 {
			out.Usage.InputTokensDetails = &ResponsesInputTokensDetails{
				CachedTokens: resp.Usage.PromptTokensDetails.CachedTokens,
			}
		}
	}

	return out
}

// extractContentText extracts the plain text from a Chat Completions content
// field (which is either a JSON string or an array of content parts).
func extractContentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []ResponsesContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "input_text", "output_text", "text":
			b.WriteString(p.Text)
		case "reasoning_text":
			// skip — reasoning is extracted separately
		}
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// Streaming: ChatCompletionsChunk → []ResponsesStreamEvent (stateful converter)
// ---------------------------------------------------------------------------

// CCEventToResponsesState tracks state for converting a sequence of
// Chat Completions SSE chunks into Responses SSE events.
type CCEventToResponsesState struct {
	ResponseID string
	Model      string
	Created    int64

	CreatedSent   bool
	CompletedSent bool

	SequenceNumber int
	OutputIndex    int
	CurrentItemID   string
	CurrentItemType string // "message" | "function_call" | "reasoning"
	CurrentCallID   string // for function_call items

	// Tool call tracking by CC index
	currentToolIndex     int
	currentToolCallID    string
	currentToolCallName  string
	pendingToolCallStart bool   // received name/id but not yet emitted
	pendingToolCallName  string // the function name for the pending tool

	// Accumulated output for response.completed
	accumulatedTextBuf     strings.Builder
	persistentReasoningBuf strings.Builder // survives reasoning-item close, used for message content embedding
	accumulatedArgsBuf      strings.Builder
	accumulatedOutputs      []ResponsesOutput

	// Usage
	InputTokens          int
	OutputTokens         int
	CacheReadInputTokens int
}

// NewCCEventToResponsesState returns an initialised stream state.
func NewCCEventToResponsesState() *CCEventToResponsesState {
	return &CCEventToResponsesState{
		ResponseID:        generateResponsesID(),
		Created:           time.Now().Unix(),
		currentToolIndex:  -1,
	}
}

// CCEventToResponsesEvents converts a single CC SSE chunk into zero or more
// Responses SSE events, updating state as it goes.
func CCEventToResponsesEvents(chunk *ChatCompletionsChunk, state *CCEventToResponsesState) []ResponsesStreamEvent {
	if state.Model == "" && chunk.Model != "" {
		state.Model = chunk.Model
	}

	var events []ResponsesStreamEvent

	// Process usage from any chunk (DeepSeek embeds usage in the final chunk).
	if chunk.Usage != nil {
		state.InputTokens = chunk.Usage.PromptTokens
		state.OutputTokens = chunk.Usage.CompletionTokens
		if chunk.Usage.PromptTokensDetails != nil {
			state.CacheReadInputTokens = chunk.Usage.PromptTokensDetails.CachedTokens
		}
	}
	// Empty choices with usage only (standard OpenAI usage chunk)
	if len(chunk.Choices) == 0 {
		return nil
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	// Role chunk only emits response.created. The message item
	// is created on the first non-empty content delta so that
	// Codex has a single clean active-item transition.
	if delta.Role != "" && !state.CreatedSent {
		events = append(events, emitCCResponsesCreated(state))
	}

	// Content delta
	if delta.Content != nil && *delta.Content != "" {
		if state.CurrentItemType != "message" {
			if !state.CreatedSent {
				events = append(events, emitCCResponsesCreated(state))
			}
			events = append(events, emitCCMessageItemAdded(state))
		}
		state.accumulatedTextBuf.WriteString(*delta.Content)
		events = append(events, makeCCEvent(state, "response.output_text.delta", &ResponsesStreamEvent{
			OutputIndex: state.OutputIndex,
			Delta:       *delta.Content,
			ItemID:      state.CurrentItemID,
		}))
	}

	// Reasoning content delta — emit streaming events for Codex display.
	// Reasoning text is also buffered for the output_item.done payload.
	if delta.ReasoningContent != nil && *delta.ReasoningContent != "" {
		if state.CurrentItemType != "reasoning" {
			if !state.CreatedSent {
				events = append(events, emitCCResponsesCreated(state))
			}
			// Close any previous item
			events = append(events, closeCCCurrentResponsesItem(state)...)
			// Open reasoning item for streaming display
			state.CurrentItemID = generateItemID()
			state.CurrentItemType = "reasoning"
			events = append(events, makeCCEvent(state, "response.output_item.added", &ResponsesStreamEvent{
				OutputIndex: state.OutputIndex,
				Item: &ResponsesOutput{
					Type:             "reasoning",
					ID:               state.CurrentItemID,
					EncryptedContent: "",
					Content:          make([]ResponsesContentPart, 0),
					Summary: []ResponsesSummary{
						{Type: "summary_text", Text: ""},
					},
				},
			}))
		}
		state.persistentReasoningBuf.WriteString(*delta.ReasoningContent)
		events = append(events, makeCCEvent(state, "response.reasoning_summary_text.delta", &ResponsesStreamEvent{
			OutputIndex:  state.OutputIndex,
			SummaryIndex: 0,
			Delta:        *delta.ReasoningContent,
			ItemID:       state.CurrentItemID,
		}))
		events = append(events, makeCCEvent(state, "response.reasoning_text.delta", &ResponsesStreamEvent{
			OutputIndex:  state.OutputIndex,
			ContentIndex: 0,
			Delta:        *delta.ReasoningContent,
			ItemID:       state.CurrentItemID,
		}))
	}

	// Tool calls
	for _, tc := range delta.ToolCalls {
		if tc.Index != nil {
			idx := *tc.Index
			if idx != state.currentToolIndex {
				// New tool call starting
				state.currentToolIndex = idx
				state.pendingToolCallStart = true
			}
		}

		if tc.ID != "" {
			state.currentToolCallID = tc.ID
		}
		if tc.Function.Name != "" {
			state.pendingToolCallName = tc.Function.Name
		}

		if state.pendingToolCallStart && state.currentToolCallID != "" && state.pendingToolCallName != "" {
			// Close current item if any
			events = append(events, closeCCCurrentResponsesItem(state)...)

			state.accumulatedArgsBuf.Reset()
			state.CurrentItemID = generateItemID()
			state.currentToolCallName = state.pendingToolCallName
			state.CurrentCallID = state.currentToolCallID

			// Map known tool names to native Responses types.
			itemType := "function_call"
			if state.currentToolCallName == "web_search" {
				itemType = "web_search_call"
			}
			state.CurrentItemType = itemType

			events = append(events, makeCCEvent(state, "response.output_item.added", &ResponsesStreamEvent{
				OutputIndex: state.OutputIndex,
				Item: &ResponsesOutput{
					Type:   itemType,
					ID:     state.CurrentItemID,
					CallID: state.currentToolCallID,
					Name:   state.currentToolCallName,
					Status: "in_progress",
				},
			}))

			state.pendingToolCallStart = false
			state.pendingToolCallName = ""
		}

		if tc.Function.Arguments != "" {
			state.accumulatedArgsBuf.WriteString(tc.Function.Arguments)
			events = append(events, makeCCEvent(state, "response.function_call_arguments.delta", &ResponsesStreamEvent{
				OutputIndex: state.OutputIndex,
				Delta:       tc.Function.Arguments,
				ItemID:      state.CurrentItemID,
				CallID:      state.currentToolCallID,
				Name:        state.currentToolCallName,
			}))
		}
	}

	// Finish reason
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		events = append(events, finishCCResponsesStream(state, *choice.FinishReason)...)
	}

	return events
}

// FinalizeCCToResponsesStream emits synthetic termination events if the stream
// ended without a proper finish_reason chunk.
func FinalizeCCToResponsesStream(state *CCEventToResponsesState) []ResponsesStreamEvent {
	if !state.CreatedSent || state.CompletedSent {
		return nil
	}
	return finishCCResponsesStream(state, "stop")
}

// --- internal streaming helpers ---

func emitCCResponsesCreated(state *CCEventToResponsesState) ResponsesStreamEvent {
	state.CreatedSent = true
	state.SequenceNumber++
	return ResponsesStreamEvent{
		Type:           "response.created",
		SequenceNumber: state.SequenceNumber - 1,
		Response: &ResponsesResponse{
			ID:     state.ResponseID,
			Object: "response",
			Model:  state.Model,
			Status: "in_progress",
			Output: []ResponsesOutput{},
		},
	}
}

func emitCCMessageItemAdded(state *CCEventToResponsesState) ResponsesStreamEvent {
	state.CurrentItemID = generateItemID()
	state.CurrentItemType = "message"
	state.accumulatedTextBuf.Reset()

	state.SequenceNumber++
	return ResponsesStreamEvent{
		Type:           "response.output_item.added",
		SequenceNumber: state.SequenceNumber - 1,
		OutputIndex:    state.OutputIndex,
		Item: &ResponsesOutput{
			Type:    "message",
			ID:      state.CurrentItemID,
			Role:    "assistant",
			Status:  "in_progress",
			Content: make([]ResponsesContentPart, 0),
		},
	}
}

func finishCCResponsesStream(state *CCEventToResponsesState, finishReason string) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Close current item if any
	events = append(events, closeCCCurrentResponsesItem(state)...)

	// Map finish reason to status
	status := "completed"
	var incompleteDetails *ResponsesIncompleteDetails
	if finishReason == "length" {
		status = "incomplete"
		incompleteDetails = &ResponsesIncompleteDetails{Reason: "max_output_tokens"}
	}

	usage := &ResponsesUsage{
		InputTokens:  state.InputTokens,
		OutputTokens: state.OutputTokens,
		TotalTokens:  state.InputTokens + state.OutputTokens,
	}
	if state.CacheReadInputTokens > 0 {
		usage.InputTokensDetails = &ResponsesInputTokensDetails{
			CachedTokens: state.CacheReadInputTokens,
		}
	}

	state.CompletedSent = true
	state.SequenceNumber++
	resp := &ResponsesResponse{
		ID:                state.ResponseID,
		Object:            "response",
		Model:             state.Model,
		Status:            status,
		Output:            state.accumulatedOutputs,
		Usage:             usage,
		IncompleteDetails: incompleteDetails,
	}
	if finishReason == "tool_calls" {
		endTurn := false
		resp.EndTurn = &endTurn
	}
	events = append(events, ResponsesStreamEvent{
		Type:           "response.completed",
		SequenceNumber: state.SequenceNumber - 1,
		Response: resp,
	})

	return events
}

func closeCCCurrentResponsesItem(state *CCEventToResponsesState) []ResponsesStreamEvent {
	if state.CurrentItemType == "" {
		return nil
	}

	itemType := state.CurrentItemType
	itemID := state.CurrentItemID

	var events []ResponsesStreamEvent

	switch itemType {
	case "message":
		// Build accumulated output with reasoning embedded as a
		// content part so it survives the Codex CLI round-trip.
		text := state.accumulatedTextBuf.String()
		reasoningText := state.persistentReasoningBuf.String()
		contentParts := buildMessageContentParts(reasoningText, text)
		built := ResponsesOutput{
			Type:    "message",
			ID:      itemID,
			Role:    "assistant",
			Status:  "completed",
			Content: contentParts,
		}
		state.accumulatedOutputs = append(state.accumulatedOutputs, built)

	case "reasoning":
		events = append(events, makeCCEvent(state, "response.reasoning_summary_text.done", &ResponsesStreamEvent{
			OutputIndex:  state.OutputIndex,
			SummaryIndex: 0,
			ItemID:       itemID,
		}))
		events = append(events, makeCCEvent(state, "response.reasoning_text.done", &ResponsesStreamEvent{
			OutputIndex:  state.OutputIndex,
			ContentIndex: 0,
			ItemID:       itemID,
		}))
		// Do NOT add a separate reasoning item to accumulatedOutputs.
		// Reasoning text is accumulated into persistentReasoningBuf and
		// will be embedded inside the next message item's content so it
		// survives the Codex CLI round-trip.

	case "function_call", "web_search_call":
		events = append(events, makeCCEvent(state, "response.function_call_arguments.done", &ResponsesStreamEvent{
			OutputIndex: state.OutputIndex,
			ItemID:      itemID,
			CallID:      state.CurrentCallID,
			Name:        state.currentToolCallName,
		}))

		built := ResponsesOutput{
			Type:      itemType,
			ID:        itemID,
			CallID:    state.CurrentCallID,
			Name:      state.currentToolCallName,
			Namespace: extractMCPNamespace(state.currentToolCallName),
			Arguments: state.accumulatedArgsBuf.String(),
			Status:    "completed",
		}
		state.accumulatedOutputs = append(state.accumulatedOutputs, built)
	}

	// Emit output_item.done with accumulated content so that
	// clients (Codex, etc.) can display the final output.
	doneItem := &ResponsesOutput{
		Type:   itemType,
		ID:     itemID,
		Status: "completed",
	}
	if itemType == "message" {
		text := state.accumulatedTextBuf.String()
		reasoningText := state.persistentReasoningBuf.String()
		doneItem.Role = "assistant"
		doneItem.Content = buildMessageContentParts(reasoningText, text)
	} else if itemType == "reasoning" {
		reasoningText := state.persistentReasoningBuf.String()
		doneItem.EncryptedContent = base64.StdEncoding.EncodeToString([]byte(reasoningText))
		doneItem.Content = []ResponsesContentPart{
			{Type: "reasoning_text", Text: reasoningText},
		}
		doneItem.Summary = []ResponsesSummary{
			{Type: "summary_text", Text: reasoningText},
		}
	} else if itemType == "function_call" || itemType == "web_search_call" {
		doneItem.CallID = state.CurrentCallID
		doneItem.Name = state.currentToolCallName
		doneItem.Namespace = extractMCPNamespace(state.currentToolCallName)
		doneItem.Arguments = state.accumulatedArgsBuf.String()
	}
	events = append(events, makeCCEvent(state, "response.output_item.done", &ResponsesStreamEvent{
		OutputIndex: state.OutputIndex,
		Item:       doneItem,
	}))

	// Clear persistent reasoning after a message item closes.
	// Reasoning items intentionally persist for the next message.
	if itemType == "message" {
		state.persistentReasoningBuf.Reset()
	}

	// Reset
	state.CurrentItemType = ""
	state.CurrentItemID = ""
	state.CurrentCallID = ""
	state.currentToolCallName = ""
	state.OutputIndex++

	return events
}

func makeCCEvent(state *CCEventToResponsesState, eventType string, template *ResponsesStreamEvent) ResponsesStreamEvent {
	state.SequenceNumber++
	template.Type = eventType
	template.SequenceNumber = state.SequenceNumber - 1
	return *template
}

// buildMessageContentParts returns a content parts slice for a message output
// item. Reasoning is NOT embedded as reasoning_text in the content because
// reasoning_text is a valid output-only type — including it in message content
// causes "unknown variant reasoning_text" errors when it round-trips through
// a client back into the input array. Reasoning is emitted as a separate
// reasoning output item instead.
func buildMessageContentParts(_, responseText string) []ResponsesContentPart {
	var parts []ResponsesContentPart
	if strings.TrimSpace(responseText) != "" {
		parts = append(parts, ResponsesContentPart{
			Type: "output_text",
			Text: responseText,
		})
	}
	if len(parts) == 0 {
		parts = append(parts, ResponsesContentPart{
			Type: "output_text",
			Text: "",
		})
	}
	return parts
}

// CCChunkToResponsesSSE formats a ResponsesStreamEvent as an SSE data line
// specifically for the Responses SSE protocol (with event: type prefix).
func CCChunkToResponsesSSE(evt ResponsesStreamEvent) (string, error) {
	data, err := json.Marshal(evt)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("event: %s\ndata: %s\n\n", evt.Type, data), nil
}

// extractMCPNamespace returns the MCP server namespace from a function name.
// For namespace tools (mcp__server) it returns the full name.
// For individual tools (mcp__server__tool) it returns mcp__server.
// Returns "" for non-MCP names.
func extractMCPNamespace(name string) string {
	if !strings.HasPrefix(name, "mcp__") {
		return ""
	}
	first := strings.Index(name, "__")
	last := strings.LastIndex(name, "__")
	if last > first {
		return name[:last]
	}
	return name
}
