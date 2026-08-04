import * as vscode from 'vscode';
import { GuardBridge } from './guardBridge';
import { createProvider } from './providers';
import { AgentSession, AgentEvent } from './agentLoop';

export class ChatViewProvider implements vscode.WebviewViewProvider {
  private view?: vscode.WebviewView;
  private session?: AgentSession;
  private guard?: GuardBridge;

  constructor(private readonly extensionUri: vscode.Uri) {}

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = this.html(webviewView.webview);

    webviewView.webview.onDidReceiveMessage(async (msg) => {
      if (msg.type !== 'send') return;
      try {
        const session = this.ensureSession();
        await session.send(msg.text);
      } catch (e: any) {
        this.post({ type: 'error', message: e?.message ?? String(e) });
      }
    });

    webviewView.onDidDispose(() => this.guard?.dispose());
  }

  private post(e: AgentEvent): void {
    this.view?.webview.postMessage(e);
  }

  private ensureSession(): AgentSession {
    if (this.session) return this.session;

    const cfg = vscode.workspace.getConfiguration('clarusAgent');
    const providerKind = cfg.get<'anthropic' | 'openai'>('provider', 'anthropic');
    const apiKey =
      providerKind === 'openai'
        ? cfg.get<string>('openaiApiKey', '') || process.env.OPENAI_API_KEY || ''
        : cfg.get<string>('anthropicApiKey', '') || process.env.ANTHROPIC_API_KEY || '';
    if (!apiKey) {
      throw new Error(
        `No API key configured for provider "${providerKind}". Set clarusAgent.${providerKind}ApiKey ` +
          `in settings, or the ${providerKind === 'openai' ? 'OPENAI_API_KEY' : 'ANTHROPIC_API_KEY'} env var.`,
      );
    }
    const model =
      providerKind === 'openai'
        ? cfg.get<string>('openaiModel', 'gpt-4.1')
        : cfg.get<string>('anthropicModel', 'claude-sonnet-5');

    const pythonPath = cfg.get<string>('pythonPath', 'python');
    const guardRepoPath =
      cfg.get<string>('guardRepoPath', '') || vscode.Uri.joinPath(this.extensionUri, '..').fsPath;
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? guardRepoPath;

    this.guard = new GuardBridge(pythonPath, guardRepoPath);
    const provider = createProvider(providerKind, apiKey, model);

    this.session = new AgentSession(
      provider,
      this.guard,
      workspaceRoot,
      (name, args) => this.requestApproval(name, args),
      (e) => this.post(e),
    );
    return this.session;
  }

  private async requestApproval(name: string, args: Record<string, unknown>): Promise<boolean> {
    const choice = await vscode.window.showWarningMessage(
      `Clarus Agent wants to run "${name}" with ${JSON.stringify(args)}. Approve?`,
      { modal: true },
      'Approve',
      'Deny',
    );
    return choice === 'Approve';
  }

  private html(webview: vscode.Webview): string {
    const css = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'main.css'));
    const js = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'main.js'));
    return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <link rel="stylesheet" href="${css}" />
</head>
<body>
  <div id="log"></div>
  <div id="inputRow">
    <textarea id="input" rows="2" placeholder="Ask Clarus Agent to read, write, or run something..."></textarea>
    <button id="send">Send</button>
  </div>
  <script src="${js}"></script>
</body>
</html>`;
  }
}
