let chatHistory = [];
const DEPLOYED_URL = "https://ask-this-page-1.onrender.com";

function addMessageToChat(sender, text) {
  const chatWindow = document.getElementById('chat-window');
  const messageDiv = document.createElement('div');
  messageDiv.className = sender === 'user' ? 'message user-message' : 'message ai-message';
  messageDiv.innerHTML = text.replace(/\n/g, '<br>').replace(/\* /g, '• ');
  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return messageDiv;
}

function fetchAndProcess(endpoint, body) {
  const statusDiv = document.getElementById('status');
  const askButton = document.getElementById('askButton');

  fetch(`${DEPLOYED_URL}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  .then(async (response) => {
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      throw new Error('Server is waking up. Please wait 30-60 seconds and try again.');
    }
    
    const data = await response.json();
    if (data.error) throw new Error(data.error);
    
    statusDiv.textContent = 'Status: Ready to chat!';
    askButton.disabled = false;
    addMessageToChat('ai', "✅ I've finished reading. Ask me anything!");
  })
  .catch(error => {
    statusDiv.textContent = 'Error: Server not ready';
    addMessageToChat('ai', `❌ ${error.message}`);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const processButton = document.getElementById('processButton');
  const askButton = document.getElementById('askButton');
  const questionInput = document.getElementById('questionInput');

  processButton.addEventListener('click', () => {
    document.getElementById('status').textContent = 'Waking up server...';
    chatHistory = [];
    document.getElementById('chat-window').innerHTML = '';

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const url = tabs[0].url;

      if (url.includes("youtube.com/watch")) {
        const videoId = new URLSearchParams(new URL(url).search).get('v');
        fetchAndProcess('/process_youtube', { videoId });
      } else if (url.endsWith(".pdf")) {
        fetchAndProcess('/process_pdf', { pdf_url: url });
      } else {
        chrome.scripting.executeScript({
          target: { tabId: tabs[0].id },
          func: () => document.documentElement.outerHTML,
        }, (results) => {
          fetchAndProcess('/process_webpage', { content: results[0].result });
        });
      }
    });
  });

  askButton.addEventListener('click', () => {
    const question = questionInput.value.trim();
    if (!question) return;

    addMessageToChat('user', question);
    const thinking = addMessageToChat('ai', 'Thinking...');
    questionInput.value = '';

    fetch(`${DEPLOYED_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history: chatHistory }),
    })
    .then(res => res.json())
    .then(data => {
      thinking.remove();
      addMessageToChat('ai', data.answer);
      chatHistory.push({ type: 'human', content: question }, { type: 'ai', content: data.answer });
    });
  });
});