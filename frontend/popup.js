let chatHistory = [];
const DEPLOYED_URL = "http://127.0.0.1:5000/"; // Ensure trailing slash

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
  const questionInput = document.getElementById('questionInput');

  // Remove leading slash if it exists to prevent double slashes in URL
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint.substring(1) : endpoint;

  fetch(`${DEPLOYED_URL}${cleanEndpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  .then(async (response) => {
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      throw new Error('Server returned invalid data. Check backend logs.');
    }
    
    const data = await response.json();
    if (data.error) throw new Error(data.error);
    
    statusDiv.textContent = 'Status: Ready to chat!';
    askButton.disabled = false;
    questionInput.disabled = false; 
    addMessageToChat('ai', "✅ I've finished reading. Ask me anything!");
  })
  .catch(error => {
    statusDiv.textContent = 'Error: Processing failed';
    addMessageToChat('ai', `❌ ${error.message}`);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const processButton = document.getElementById('processButton');
  const askButton = document.getElementById('askButton');
  const questionInput = document.getElementById('questionInput');

  processButton.addEventListener('click', () => {
    document.getElementById('status').textContent = 'Processing page...';
    chatHistory = [];
    document.getElementById('chat-window').innerHTML = '';
    
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const url = tabs[0].url;

      if (url.includes("youtube.com/watch")) {
        const videoId = new URLSearchParams(new URL(url).search).get('v');
        fetchAndProcess('process_youtube', { videoId });
      } else if (url.endsWith(".pdf")) {
        fetchAndProcess('process_pdf', { pdf_url: url });
      } else {
        chrome.scripting.executeScript({
          target: { tabId: tabs[0].id },
          // Sending clean text instead of heavy HTML
          func: () => document.body.innerText,
        }, (results) => {
          if(results && results[0]) {
             fetchAndProcess('process_webpage', { content: results[0].result });
          } else {
             document.getElementById('status').textContent = 'Error: Could not read page text.';
          }
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

    fetch(`${DEPLOYED_URL}ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history: chatHistory }),
    })
    .then(res => res.json())
    .then(data => {
      thinking.remove();
      if (data.error) {
         addMessageToChat('ai', `❌ Error: ${data.error}`);
      } else {
         addMessageToChat('ai', data.answer);
         chatHistory.push({ type: 'human', content: question }, { type: 'ai', content: data.answer });
      }
    })
    .catch(error => {
       thinking.remove();
       addMessageToChat('ai', `❌ Error: ${error.message}`);
    });
  });
});