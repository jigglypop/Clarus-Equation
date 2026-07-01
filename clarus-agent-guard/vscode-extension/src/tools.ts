// Tool catalogue shared by every provider. Only the shape is defined here;
// the real side effects live behind server/vscode_bridge.py (guardBridge.ts
// talks to it) so every side-effecting call passes through ClarusGuard.

export interface ToolDef {
  name: string;
  description: string;
  sideEffecting: boolean;
  parameters: {
    type: 'object';
    properties: Record<string, { type: string; description: string }>;
    required: string[];
  };
}

export const TOOL_DEFS: ToolDef[] = [
  {
    name: 'read_file',
    description: 'Read a UTF-8 text file at a workspace-relative or absolute path.',
    sideEffecting: false,
    parameters: {
      type: 'object',
      properties: { path: { type: 'string', description: 'File path to read' } },
      required: ['path'],
    },
  },
  {
    name: 'list_dir',
    description: 'List entries of a directory.',
    sideEffecting: false,
    parameters: {
      type: 'object',
      properties: { path: { type: 'string', description: 'Directory path to list' } },
      required: ['path'],
    },
  },
  {
    name: 'write_file',
    description: 'Write (create or overwrite) a UTF-8 text file. Side-effecting: requires user approval.',
    sideEffecting: true,
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path to write' },
        content: { type: 'string', description: 'Full file content to write' },
      },
      required: ['path', 'content'],
    },
  },
  {
    name: 'run_command',
    description: 'Run a shell command in the workspace root. Side-effecting: requires user approval.',
    sideEffecting: true,
    parameters: {
      type: 'object',
      properties: { command: { type: 'string', description: 'Shell command to execute' } },
      required: ['command'],
    },
  },
];

export function toolDef(name: string): ToolDef | undefined {
  return TOOL_DEFS.find((t) => t.name === name);
}
