(function () {
  const vscode = acquireVsCodeApi();
  const log = document.getElementById('log');
  const input = document.getElementById('input');
  const send = document.getElementById('send');

  function append(cls, text) {
    const div = document.createElement('div');
    div.className = 'msg ' + cls;
    div.textContent = text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  send.addEventListener('click', submit);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });

  function submit() {
    const text = input.value.trim();
    if (!text) return;
    append('user', text);
    vscode.postMessage({ type: 'send', text });
    input.value = '';
  }

  window.addEventListener('message', (event) => {
    const m = event.data;
    if (m.type === 'assistant_text') append('assistant', m.text);
    else if (m.type === 'tool_call') append('tool', `-> ${m.name}(${JSON.stringify(m.args)})`);
    else if (m.type === 'tool_result') append('tool', `<- ${m.name}: ${m.status}${m.reason ? ' — ' + m.reason : ''}`);
    else if (m.type === 'error') append('error', m.message);
  });
})();
