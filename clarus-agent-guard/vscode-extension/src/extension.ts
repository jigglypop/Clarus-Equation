import * as vscode from 'vscode';
import { ChatViewProvider } from './chatViewProvider';

export function activate(context: vscode.ExtensionContext): void {
  const provider = new ChatViewProvider(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('clarusAgent.chat', provider),
    vscode.commands.registerCommand('clarusAgent.newSession', () => {
      vscode.commands.executeCommand('workbench.view.extension.clarusAgent');
    }),
  );
}

export function deactivate(): void {}
