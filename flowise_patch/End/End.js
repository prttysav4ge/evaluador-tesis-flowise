"use strict";
// End node for Flowise Agentflow
// Terminates the agentflow and returns the selected output.
// Generated as a patch because this component is missing from the installed
// flowiseai/flowise Docker image (tested on flowise v3.1.2).
//
// For "returnLastOutput":
//   Primary source: agentflowRuntime.chatHistory (accumulated by the execution engine
//   at buildAgentflow.js:1564 — always contains the last node's response as the
//   final element).
//   Fallback: last non-empty, non-template value in agentflowRuntime.state.
//
// For "returnAllStateValues":
//   Return JSON of the entire flow state.
//
// For "returnCustomStateValues":
//   Return JSON of the selected state keys.
Object.defineProperty(exports, "__esModule", { value: true });

class End_Agentflow {
    constructor() {
        this.label = 'End';
        this.name = 'endAgentflow';
        this.version = 1.1;
        this.type = 'End';
        this.color = '#FF8A80';
        this.hideOutput = true;
        this.baseClasses = ['End'];
        this.category = 'Agent Flows';
        this.description = 'End point of the agentflow';
        this.inputs = [];
        this.inputParams = [
            {
                label: 'End Output',
                name: 'endOutput',
                type: 'options',
                options: [
                    {
                        label: 'Return All State Values',
                        name: 'returnAllStateValues',
                        description: 'Return all state values at the end of the workflow'
                    },
                    {
                        label: 'Return Last Output',
                        name: 'returnLastOutput',
                        description: 'Return the last output from the workflow'
                    },
                    {
                        label: 'Return Custom State Values',
                        name: 'returnCustomStateValues',
                        description: 'Specify which state values to return'
                    }
                ],
                default: 'returnLastOutput'
            },
            {
                label: 'Custom State Values',
                name: 'endCustomStateValues',
                type: 'array',
                show: { endOutput: 'returnCustomStateValues' },
                optional: true,
                array: [
                    {
                        label: 'Key',
                        name: 'key',
                        type: 'asyncOptions',
                        loadMethod: 'listRuntimeStateKeys'
                    }
                ]
            }
        ];
        this.inputAnchors = [];
        this.outputAnchors = [];
    }

    /**
     * Extract plain-text content from a chat message's content field.
     * LangChain messages can have content as string or array of content parts.
     */
    _extractMessageContent(content) {
        if (typeof content === 'string') return content;
        if (Array.isArray(content)) {
            return content
                .map(part => (typeof part === 'string' ? part : (part?.text || part?.content || '')))
                .join('');
        }
        if (content && typeof content === 'object') {
            return content.text || content.content || JSON.stringify(content);
        }
        return String(content || '');
    }

    async run(nodeData, input, options) {
        const endOutput = nodeData.inputs?.endOutput || 'returnLastOutput';
        const state = options?.agentflowRuntime?.state || {};

        let content = '';

        if (endOutput === 'returnLastOutput') {
            // ── Primary: last message in the accumulated chat history ──────────
            // The execution engine appends each node's chatHistory to
            // agentflowRuntime.chatHistory (buildAgentflow.js line ~1564).
            // The very last element is always the most-recently-executed node's output.
            const chatHistory = options?.agentflowRuntime?.chatHistory || [];
            if (chatHistory.length > 0) {
                const lastMsg = chatHistory[chatHistory.length - 1];
                if (lastMsg) {
                    const extracted = this._extractMessageContent(lastMsg.content);
                    if (extracted.trim()) {
                        content = extracted;
                    }
                }
            }

            // ── Fallback: state (last meaningful, non-template value) ──────────
            if (!content) {
                const stateKeys = Object.keys(state);
                for (let i = stateKeys.length - 1; i >= 0; i--) {
                    const val = state[stateKeys[i]];
                    if (val === null || val === undefined) continue;
                    const strVal = String(val);
                    if (strVal.trim() === '') continue;
                    // Skip unresolved template variables like {{ $returnValue.xxx }}
                    if (/^\s*\{\{[^}]+\}\}\s*$/.test(strVal)) continue;
                    // Skip trivially empty JSON
                    if (strVal.trim() === '{}' || strVal.trim() === '[]') continue;
                    content = strVal;
                    break;
                }
            }

            // ── Last resort: original question ────────────────────────────────
            if (!content) {
                content = typeof input === 'string' ? input : '';
            }

        } else if (endOutput === 'returnAllStateValues') {
            content = JSON.stringify(state, null, 2);

        } else if (endOutput === 'returnCustomStateValues') {
            const customKeys = nodeData.inputs?.endCustomStateValues || [];
            const result = {};
            for (const item of customKeys) {
                // item may be a string key or an object { key: "...", ... }
                const key = typeof item === 'string' ? item : item?.key;
                if (key && state[key] !== undefined) {
                    result[key] = state[key];
                }
            }
            content = JSON.stringify(result, null, 2);
        }

        // Strip residual HTML that was not processed by resolveVariables
        // (only strip when the string clearly starts with an HTML block element)
        if (content && /^\s*<(?:p|div|h[1-6]|ul|ol|blockquote|pre|table)\b/i.test(content)) {
            content = content.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
        }

        return {
            id: nodeData.id,
            name: this.name,
            input: {},
            output: { content },
            state
        };
    }
}

module.exports = { nodeClass: End_Agentflow };
//# sourceMappingURL=End.js.map
