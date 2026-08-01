import * as cp from 'child_process';
import * as readline from 'readline';

export interface GuardResult {
  status: 'executed' | 'refused' | 'pending';
  reason: string;
  value: unknown;
  token: string;
}

// Talks to `python -m server.vscode_bridge`, a long-lived process that
// executes ClarusGuard tool calls one JSON line at a time. The bridge
// replies in the same order it receives requests, so a FIFO queue of
// resolvers is enough to pair responses with callers.
export class GuardBridge {
  private proc: cp.ChildProcessWithoutNullStreams;
  private pending: Array<(r: GuardResult) => void> = [];
  private stderrBuf = '';

  constructor(pythonPath: string, guardRepoPath: string) {
    this.proc = cp.spawn(pythonPath, ['-m', 'server.vscode_bridge'], {
      cwd: guardRepoPath,
      env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
    });
    const rl = readline.createInterface({ input: this.proc.stdout });
    rl.on('line', (line) => {
      const resolve = this.pending.shift();
      if (!resolve) return;
      try {
        resolve(JSON.parse(line));
      } catch {
        resolve({ status: 'refused', reason: `malformed bridge response: ${line}`, value: null, token: '' });
      }
    });
    this.proc.stderr.on('data', (d) => { this.stderrBuf += d.toString(); });
  }

  dispose(): void {
    this.proc.kill();
  }

  private send(req: unknown): Promise<GuardResult> {
    return new Promise((resolve) => {
      this.pending.push(resolve);
      this.proc.stdin.write(JSON.stringify(req) + '\n');
    });
  }

  call(tool: string, args: Record<string, unknown>, userText: string, provenance: 'user' | 'tool' = 'user'): Promise<GuardResult> {
    return this.send({ tool, args, user_text: userText, provenance });
  }

  approve(token: string): Promise<GuardResult> {
    return this.send({ approve: token });
  }

  lastStderr(): string {
    return this.stderrBuf;
  }
}
