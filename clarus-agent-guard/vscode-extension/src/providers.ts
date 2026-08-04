import { TOOL_DEFS } from './tools';

export interface ToolCall { id: string; name: string; args: Record<string, unknown>; }
export interface AgentTurn { text: string; toolCalls: ToolCall[]; }
export interface ToolResult { id: string; name: string; resultText: string; }

export interface LlmProvider {
  addUserMessage(text: string): void;
  addToolResults(results: ToolResult[]): void;
  step(): Promise<AgentTurn>;
}

const SYSTEM_PROMPT =
  'You are a coding agent embedded in VS Code. Use the available tools to read, ' +
  'write and run things in the user workspace. Every side-effecting tool call ' +
  '(write_file, run_command) is intercepted by a security guard and requires ' +
  'explicit human approval before it executes — expect some calls to come back ' +
  'as "pending approval" or "refused"; explain what happened to the user rather ' +
  'than silently retrying.';

function safeJson(v: unknown): string {
  try { return JSON.stringify(v); } catch { return String(v); }
}

// --- Anthropic (Claude Messages API) --------------------------------------

export class AnthropicProvider implements LlmProvider {
  private history: any[] = [];

  constructor(private apiKey: string, private model: string) {}

  addUserMessage(text: string): void {
    this.history.push({ role: 'user', content: text });
  }

  addToolResults(results: ToolResult[]): void {
    this.history.push({
      role: 'user',
      content: results.map((r) => ({
        type: 'tool_result',
        tool_use_id: r.id,
        content: r.resultText,
      })),
    });
  }

  async step(): Promise<AgentTurn> {
    const tools = TOOL_DEFS.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.parameters,
    }));

    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': this.apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: this.model,
        max_tokens: 4096,
        system: SYSTEM_PROMPT,
        tools,
        messages: this.history,
      }),
    });
    if (!res.ok) {
      throw new Error(`Anthropic API error ${res.status}: ${await res.text()}`);
    }
    const data: any = await res.json();
    this.history.push({ role: 'assistant', content: data.content });

    let text = '';
    const toolCalls: ToolCall[] = [];
    for (const block of data.content ?? []) {
      if (block.type === 'text') text += block.text;
      if (block.type === 'tool_use') {
        toolCalls.push({ id: block.id, name: block.name, args: block.input ?? {} });
      }
    }
    return { text, toolCalls };
  }
}

// --- OpenAI (Chat Completions API, function calling) ----------------------

export class OpenAiProvider implements LlmProvider {
  private history: any[] = [{ role: 'system', content: SYSTEM_PROMPT }];

  constructor(private apiKey: string, private model: string) {}

  addUserMessage(text: string): void {
    this.history.push({ role: 'user', content: text });
  }

  addToolResults(results: ToolResult[]): void {
    for (const r of results) {
      this.history.push({ role: 'tool', tool_call_id: r.id, content: r.resultText });
    }
  }

  async step(): Promise<AgentTurn> {
    const tools = TOOL_DEFS.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));

    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ model: this.model, messages: this.history, tools }),
    });
    if (!res.ok) {
      throw new Error(`OpenAI API error ${res.status}: ${await res.text()}`);
    }
    const data: any = await res.json();
    const msg = data.choices[0].message;
    this.history.push(msg);

    const toolCalls: ToolCall[] = (msg.tool_calls ?? []).map((tc: any) => ({
      id: tc.id,
      name: tc.function.name,
      args: (() => {
        try { return JSON.parse(tc.function.arguments || '{}'); } catch { return {}; }
      })(),
    }));
    return { text: msg.content ?? '', toolCalls };
  }
}

export function createProvider(kind: 'anthropic' | 'openai', apiKey: string, model: string): LlmProvider {
  if (kind === 'openai') return new OpenAiProvider(apiKey, model);
  return new AnthropicProvider(apiKey, model);
}

export { safeJson };
