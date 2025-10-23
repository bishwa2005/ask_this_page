let chatHistory = [];

/**
 * Adds a message to the chat UI and handles Markdown/Copy button.
 * @param {string} sender - 'user' or 'ai'
 * @param {string} text - The raw text of the message
 */
function addMessageToChat(sender, text) {
  const chatWindow = document.getElementById('chat-window');
  
  // Remove empty state if it exists
  const emptyState = chatWindow.querySelector('.empty-state');
  if (emptyState) {
    emptyState.remove();
  }
  
  const messageDiv = document.createElement('div');
  messageDiv.className = sender === 'user' ? 'message user-message' : 'message ai-message';

  if (sender === 'user') {
    messageDiv.textContent = text;
  } else {
    // Check if this is a "Thinking..." message
    if (text === 'Thinking...') {
      messageDiv.className = 'message ai-message';
      messageDiv.innerHTML = `
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      `;
    } else {
      // Render Markdown
      messageDiv.innerHTML = marked.parse(text);

      // Add Copy Button
      const copyBtn = document.createElement('button');
      copyBtn.className = 'copy-btn';
      copyBtn.textContent = '📋';
      copyBtn.title = 'Copy text';
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(text).then(() => {
          copyBtn.textContent = '✅';
          setTimeout(() => { copyBtn.textContent = '📋'; }, 1500);
        }).catch(err => console.error('Failed to copy text: ', err));
      });
      messageDiv.appendChild(copyBtn);
    }
  }

  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return messageDiv;
}

/**
 * Updates the status badge
 * @param {string} status - 'processing', 'ready', 'error', or 'idle'
 * @param {string} text - Status text to display
 */
function updateStatus(status, text) {
  const statusDiv = document.getElementById('status');
  const statusText = statusDiv.querySelector('.status-text');
  
  statusDiv.className = 'status-badge ' + status;
  statusText.textContent = text;
}

/**
 * Handles sending a question to the backend and updating the UI.
 * @param {string} question - The question text to send.
 */
function sendQuestion(question) {
  if (!question) return;

  const askButton = document.getElementById('askButton');
  const questionInput = document.getElementById('questionInput');
  
  // Add user's message to UI and history
  addMessageToChat('user', question);
  chatHistory.push({ type: "human", content: question });

  // Clear input (if it was the source) and disable controls
  questionInput.value = '';
  askButton.disabled = true;
  questionInput.disabled = true;
  
  const thinkingMessage = addMessageToChat('ai', 'Thinking...');

  fetch('http://127.0.0.1:5000/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      question: question,
      history: chatHistory
    }),
  })
  .then(response => response.json())
  .then(data => {
    thinkingMessage.remove();
    if (data.error) {
      addMessageToChat('ai', `❌ Error: ${data.error}`);
    } else {
      addMessageToChat('ai', data.answer);
      chatHistory.push({ type: "ai", content: data.answer });
    }
  })
  .catch(error => {
    thinkingMessage.remove();
    addMessageToChat('ai', `❌ Error: ${error.message}`);
  })
  .finally(() => {
    askButton.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  });
}

// --- Main DOMContentLoaded Event ---
document.addEventListener('DOMContentLoaded', () => {
  // Get references to elements
  const processButton = document.getElementById('processButton');
  const askButton = document.getElementById('askButton');
  const questionInput = document.getElementById('questionInput');
  const chatWindow = document.getElementById('chat-window');
  const quickPromptsContainer = document.getElementById('quickPromptsContainer');
  const themeToggle = document.getElementById('themeToggle');

  // Load theme preference
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);

  // Theme toggle
  themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
  });

  function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('.theme-icon');
    icon.textContent = theme === 'light' ? '🌙' : '☀️';
  }

  // --- "Process Page" button listener ---
  processButton.addEventListener('click', () => {
    updateStatus('processing', 'Processing...');
    processButton.disabled = true;
    processButton.classList.add('loading');
    askButton.disabled = true;
    questionInput.disabled = true;
    quickPromptsContainer.style.display = 'none';
    chatWindow.innerHTML = '';
    chatHistory = []; 
    addMessageToChat('ai', '🔍 Figuring out what to read...');

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const activeTab = tabs[0];
      const pageUrl = activeTab.url;

      if (pageUrl.endsWith('.pdf')) {
        addMessageToChat('ai', '📄 Found a PDF! Fetching and reading...');
        fetchAndProcess('http://127.0.0.1:5000/process_pdf', { pdf_url: pageUrl });
      
      } else if (pageUrl.includes("youtube.com/watch")) {
        const urlParams = new URLSearchParams(new URL(pageUrl).search);
        const videoId = urlParams.get('v');
        if (videoId) {
          addMessageToChat('ai', '🎥 Found a YouTube video! Fetching transcript...');
          fetchAndProcess('http://127.0.0.1:5000/process_youtube', { videoId: videoId });
        } else {
          showError('Could not find YouTube video ID');
        }

      } else {
        // --- Original Webpage Logic ---
        addMessageToChat('ai', '🌐 Reading the webpage content...');
        chrome.scripting.executeScript({
            target: { tabId: activeTab.id },
            func: () => document.documentElement.outerHTML, 
        }, (injectionResults) => {
            if (chrome.runtime.lastError || !injectionResults || !injectionResults[0]) {
              showError('Could not access this page');
              return;
            }
            const pageHtml = injectionResults[0].result;
            fetchAndProcess('http://127.0.0.1:5000/process_webpage', { content: pageHtml });
        });
      }
    });
  });

  // Helper to DRY up fetch logic
  function fetchAndProcess(url, body) {
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        showError(data.error);
      } else {
        updateStatus('ready', 'Ready to answer questions!');
        askButton.disabled = false;
        questionInput.disabled = false;
        quickPromptsContainer.style.display = 'flex';
        chatWindow.innerHTML = ''; 
        addMessageToChat('ai', "✅ I've finished reading. What would you like to know?");
      }
    })
    .catch(error => {
      showError(`Cannot connect to backend server. ${error.message}`);
    })
    .finally(() => {
      processButton.disabled = false;
      processButton.classList.remove('loading');
    });
  }
  
  // Helper to show errors
  function showError(errorMessage) {
    updateStatus('error', `Error: ${errorMessage}`);
    addMessageToChat('ai', `❌ Error: ${errorMessage}`);
    processButton.disabled = false;
    processButton.classList.remove('loading');
  }

  // --- "Ask" button listener ---
  askButton.addEventListener('click', () => {
    const question = questionInput.value.trim();
    sendQuestion(question);
  });

  // Allow pressing "Enter" to send
  questionInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter' && !askButton.disabled) {
      askButton.click();
    }
  });
  
  // --- One-Click Prompt Listeners ---
  document.querySelectorAll('.quick-prompt-btn').forEach(button => {
    button.addEventListener('click', () => {
      const prompt = button.getAttribute('data-prompt');
      sendQuestion(prompt);
    });
  });
});