import * as path from 'path';
import { GuardBridge } from './guardBridge';
import { LlmProvider, ToolResult, safeJson } from './providers';
import { toolDef } from './tools';

export type AgentEvent =
  | { type: 'assistant_text'; text: string }
  | { type: 'tool_call'; name: string; args: Record<string, unknown> }
  | { type: 'tool_result'; name: string; status: string; reason: string }
  | { type: 'error'; message: string };

const MAX_TURNS = 25;

export class AgentSession {
  constructor(
    private provider: LlmProvider,
    private guard: GuardBridge,
    private workspaceRoot: string,
    private requestApproval: (name: string, args: Record<string, unknown>) => Promise<boolean>,
    private onEvent: (e: AgentEvent) => void,
  ) {}

  private resolvePath(p: string): string {
    return path.isAbsolute(p) ? p : path.join(this.workspaceRoot, p);
  }

  async send(userText: string): Promise<void> {
    this.provider.addUserMessage(userText);

    for (let turn = 0; turn < MAX_TURNS; turn++) {
      let step;
      try {
        step = await this.provider.step();
      } catch (e: any) {
        this.onEvent({ type: 'error', message: e?.message ?? String(e) });
        return;
      }

      if (step.text) this.onEvent({ type: 'assistant_text', text: step.text });
      if (step.toolCalls.length === 0) return;

      const results: ToolResult[] = [];
      for (const call of step.toolCalls) {
        this.onEvent({ type: 'tool_call', name: call.name, args: call.args });
        const def = toolDef(call.name);
        const args = { ...call.args };
        if (typeof args.path === 'string') args.path = this.resolvePath(args.path);

        let r = await this.guard.call(call.name, args, userText, 'user');

        if (r.status === 'pending') {
          const approved = def?.sideEffecting
            ? await this.requestApproval(call.name, call.args)
            : false;
          r = approved
            ? await this.guard.approve(r.token)
            : { status: 'refused', reason: 'user denied approval', value: null, token: '' };
        }

        this.onEvent({ type: 'tool_result', name: call.name, status: r.status, reason: r.reason });
        const resultText = r.status === 'executed' ? safeJson(r.value) : `[${r.status}] ${r.reason}`;
        results.push({ id: call.id, name: call.name, resultText });
      }
      this.provider.addToolResults(results);
    }
    this.onEvent({ type: 'error', message: `stopped after ${MAX_TURNS} tool-use turns` });
  }
}
