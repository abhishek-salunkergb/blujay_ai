/**
 * BluJay AI - Live Interview Copilot Overlay Frontend Controller
 * Bidirectional PyWebView Bridge & 1:1 Glass UI Event Engine
 */

(function () {
  // --- DOM References ---
  const glassQaCard = document.getElementById('glass-qa-card');
  const questionText = document.getElementById('question-text');
  const answerContent = document.getElementById('answer-content');
  const qaScrollBody = document.getElementById('qa-scroll-body');
  const generationStatus = document.getElementById('generation-status');
  const liveSubtitleText = document.getElementById('live-subtitle-text');
  const manualInputField = document.getElementById('manual-input-field');
  const timerText = document.getElementById('timer-text');
  const recDot = document.getElementById('rec-dot');
  const btnSttToggle = document.getElementById('btn-stt-toggle');
  const sttText = document.getElementById('stt-text');
  const historyCounter = document.getElementById('history-counter');
  const morePopupMenu = document.getElementById('more-popup-menu');

  // --- State Variables ---
  let isCardOpen = false;
  let isTranscribing = true;
  let isStreaming = false;
  let activeQuestion = '';
  let activeAnswer = '';
  let qaHistory = [];
  let historyIdx = -1;
  let sessionTimerSeconds = 60;
  let timerInterval = null;
  let isTeleprompterActive = false;
  let teleprompterSpeed = 1.0;
  let teleprompterRafId = null;

  // --- Safe Python Bridge Invoker ---
  function callPython(methodName, ...args) {
    if (window.pywebview && window.pywebview.api && typeof window.pywebview.api[methodName] === 'function') {
      try {
        return window.pywebview.api[methodName](...args);
      } catch (err) {
        console.error(`[PyWebView Bridge Error] ${methodName}:`, err);
      }
    } else {
      console.log(`[Bridge Simulation] ${methodName}(`, ...args, `)`);
    }
  }

  // --- Session Timer ---
  function startSessionTimer() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      sessionTimerSeconds++;
      const mins = Math.floor(sessionTimerSeconds / 60);
      const secs = (sessionTimerSeconds % 60).toString().padStart(2, '0');
      if (timerText) {
        timerText.textContent = `${mins} min (Active)`;
      }
    }, 1000);
  }
  startSessionTimer();
  callPython('copilot_ready');

  // --- Card Visibility & Window Sizing ---
  function showAnswerCard() {
    if (!isCardOpen) {
      isCardOpen = true;
      if (glassQaCard) glassQaCard.classList.remove('hidden');
      callPython('resize_window', true); // Expand ~560px
    }
  }

  window.dismissAnswerCard = function () {
    isCardOpen = false;
    if (glassQaCard) glassQaCard.classList.add('hidden');
    callPython('resize_window', false); // Compact ~140px
  };

  // --- Streaming & Render State ---
  let currentGenerationId = 0;
  let renderPending = false;
  let activeCopiedBlockIdx = -1;
  let activeCopiedTimer = null;

  // --- Render-Safe Projection (Raw activeAnswer remains strictly unmutated) ---
  function getRenderSafeAnswer(text) {
    if (!text) return '';
    let renderText = text;

    // Step 0A: Incomplete triple-backtick code fence recovery for streaming view
    const fenceMatches = renderText.match(/```/g);
    if (fenceMatches && fenceMatches.length % 2 !== 0) {
      renderText += '\n```';
    }

    // Step 0B: Incomplete inline backtick recovery (outside code blocks)
    const cleanWithoutBlocks = renderText.replace(/```[\s\S]*?```/g, '');
    const singleBacktickCount = (cleanWithoutBlocks.match(/(?<!`)`(?!`)/g) || []).length;
    if (singleBacktickCount % 2 !== 0) {
      renderText += '`';
    }

    return renderText;
  }

  // --- HTML Entity Escaping Helper ---
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // --- Universal Multi-Language Single-Pass Code Highlighter ---
  function highlightCode(code, lang) {
    if (!code) return '';

    const KEYWORDS = new Set([
      // Python
      'def', 'class', 'return', 'if', 'else', 'elif', 'for', 'while', 'import', 'from', 'as', 'try', 'except', 'finally',
      'with', 'lambda', 'yield', 'pass', 'break', 'continue', 'raise', 'assert', 'global', 'nonlocal', 'del', 'async', 'await',
      'None', 'True', 'False', 'is', 'in', 'not', 'and', 'or', 'self',
      // JavaScript / TypeScript
      'const', 'let', 'var', 'function', 'export', 'default', 'catch', 'throw', 'extends', 'new', 'this', 'super', 'typeof',
      'instanceof', 'interface', 'type', 'enum', 'implements', 'null', 'undefined', 'true', 'false', 'do', 'of',
      // Java / C++ / C#
      'public', 'private', 'protected', 'static', 'final', 'void', 'switch', 'case', 'package', 'auto', 'struct', 'template',
      'typename', 'namespace', 'nullptr', 'virtual', 'override', 'constexpr', 'inline', 'bool', 'char', 'int', 'float', 'double',
      'long', 'short', 'unsigned', 'signed', 'sizeof', 'typedef', 'using', 'throws', 'synchronized', 'volatile',
      // SQL
      'select', 'from', 'where', 'group', 'by', 'order', 'having', 'join', 'inner', 'left', 'right', 'full', 'outer', 'on',
      'insert', 'into', 'values', 'update', 'set', 'delete', 'create', 'table', 'alter', 'drop', 'union', 'all', 'distinct',
      'limit', 'offset', 'case', 'when', 'then', 'end', 'asc', 'desc', 'between', 'like', 'exists', 'view', 'index'
    ]);

    const BUILTINS = new Set([
      'print', 'len', 'range', 'dict', 'list', 'set', 'tuple', 'str', 'int', 'float', 'bool', 'enumerate', 'zip', 'sorted',
      'reversed', 'min', 'max', 'sum', 'map', 'filter', 'any', 'all', 'abs', 'round', 'console', 'Math', 'Array', 'Object',
      'String', 'Number', 'Boolean', 'Promise', 'System', 'out', 'println', 'Integer', 'Double', 'Collections', 'Arrays',
      'HashMap', 'HashSet', 'ArrayList', 'vector', 'cout', 'cin', 'endl', 'dense_rank', 'rank', 'row_number', 'over',
      'partition', 'count', 'avg', 'push', 'pop', 'shift', 'unshift', 'slice', 'splice', 'indexOf', 'includes'
    ]);

    const TYPES = new Set([
      'List', 'Dict', 'Set', 'Tuple', 'Optional', 'Union', 'Any', 'Callable', 'TreeNode', 'ListNode', 'Node', 'Vector',
      'Matrix', 'Iterator', 'Iterable', 'boolean', 'byte', 'char', 'short', 'long', 'double', 'size_t', 'int64_t',
      'uint64_t', 'string', 'String', 'Integer', 'Double', 'Boolean'
    ]);

    // Tokenizer matching tokens in exact lexical precedence:
    // 1. Triple-quoted strings
    // 2. Comments (#, //, /*...*/, --)
    // 3. Quoted strings (double, single, backtick)
    // 4. Numbers
    // 5. Identifiers / Words
    // 6. Symbols, operators, and whitespace
    const tokenRegex = /("""[\s\S]*?"""|'''[\s\S]*?'''|#.*$|\/\/.*$|\/\*[\s\S]*?\*\/|--.*$|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\b\d+(?:\.\d+)?\b|\b[a-zA-Z_]\w*\b|[^\w\s]|\s+)/gm;

    let highlightedHtml = '';
    let match;
    while ((match = tokenRegex.exec(code)) !== null) {
      const token = match[0];
      if (!token) continue;

      const firstChar = token[0];
      const tokenLower = token.toLowerCase();

      // Comment token
      if (token.startsWith('#') || token.startsWith('//') || token.startsWith('/*') || token.startsWith('--')) {
        highlightedHtml += `<span class="code-comment">${escapeHtml(token)}</span>`;
      }
      // String literal token (including triple-quoted multi-line strings)
      else if (firstChar === '"' || firstChar === "'" || firstChar === '`') {
        highlightedHtml += `<span class="code-string">${escapeHtml(token)}</span>`;
      }
      // Numeric token
      else if (/^\d+(?:\.\d+)?$/.test(token)) {
        highlightedHtml += `<span class="code-number">${token}</span>`;
      }
      // Word token: Check Keywords, Builtins, Types, or Identifier
      else if (/^[a-zA-Z_]\w*$/.test(token)) {
        if (KEYWORDS.has(token) || KEYWORDS.has(tokenLower)) {
          highlightedHtml += `<span class="code-keyword">${escapeHtml(token)}</span>`;
        } else if (BUILTINS.has(token) || BUILTINS.has(tokenLower)) {
          highlightedHtml += `<span class="code-builtin">${escapeHtml(token)}</span>`;
        } else if (TYPES.has(token)) {
          highlightedHtml += `<span class="code-type">${escapeHtml(token)}</span>`;
        } else {
          highlightedHtml += escapeHtml(token);
        }
      }
      // Operators, punctuation, whitespace
      else {
        highlightedHtml += escapeHtml(token);
      }
    }

    return highlightedHtml;
  }

  // --- Syntax Highlighting & Rich Formatting ---
  function formatMarkdownAnswer(rawText) {
    if (!rawText) return '';

    let workingText = rawText;

    // Step 1: Protect and highlight code blocks before escaping/formatting
    const codeBlocks = [];
    let textWithoutCode = workingText.replace(/```([a-zA-Z0-9_\+\#-]*)\n?([\s\S]*?)```/g, function (match, lang, code) {
      const idx = codeBlocks.length;
      const id = `___CODE_BLOCK_${idx}___`;
      
      const highlightedCode = highlightCode(code, lang);

      codeBlocks.push(`
        <pre class="code-block-wrapper">
          <button class="code-copy-btn" data-block-idx="${idx}" onclick="copyCodeSnippet(this, ${idx})" title="Copy executable code">📋 Copy Code</button>
          <code>${highlightedCode}</code>
        </pre>
      `);
      return id;
    });

    // Step 2: Escape text outside code blocks
    let escaped = textWithoutCode
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Step 2B: Format inline code `code` outside blocks
    escaped = escaped.replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>');

    // Step 3: Format bullet leads, bold text, headers, and complexity metrics
    escaped = escaped.replace(/^[•\-\*]\s*\*\*(.*?)\*\*:\s*(.*)$/gim, '<p><span class="bullet-lead">• $1:</span> $2</p>');
    escaped = escaped.replace(/^[•\-\*]\s*(.*)$/gim, '<p>• $1</p>');
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    escaped = escaped.replace(/•\s*Time(?:\s*Complexity)?:\s*([^\n<]+)/gim, '<div class="complexity-badge-row"><span class="complexity-pill time-pill">⏱️ Time: $1</span></div>');
    escaped = escaped.replace(/•\s*Space(?:\s*Complexity)?:\s*([^\n<]+)/gim, '<span class="complexity-pill space-pill">💾 Space: $1</span>');

    // Step 4: Format paragraphs
    let paragraphs = escaped.split(/\n\n+/);
    let html = paragraphs.map(p => {
      if (p.startsWith('<p') || p.startsWith('___CODE_BLOCK_') || p.startsWith('<div')) return p;
      let cleanP = p.replace(/\n/g, '<br>');
      return `<p>${cleanP}</p>`;
    }).join('');

    // Step 5: Restore code blocks
    codeBlocks.forEach((cb, idx) => {
      html = html.replace(`<p>___CODE_BLOCK_${idx}___</p>`, cb);
      html = html.replace(`___CODE_BLOCK_${idx}___`, cb);
    });

    return html;
  }

  // --- Copy Code Snippet with Feedback Persistence ---
  window.copyCodeSnippet = function (btn, idx) {
    const pre = btn.closest('.code-block-wrapper');
    if (!pre) return;
    const codeEl = pre.querySelector('code');
    if (!codeEl) return;
    const codeText = codeEl.innerText || codeEl.textContent;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(codeText.trim()).catch(() => {});
    }
    callPython('copy_to_clipboard', codeText.trim());

    activeCopiedBlockIdx = typeof idx === 'number' ? idx : parseInt(btn.getAttribute('data-block-idx') || '-1', 10);
    btn.textContent = '✓ Code Copied!';
    btn.style.color = '#34d399';

    const blockIdx = activeCopiedBlockIdx;
    if (activeCopiedTimer) clearTimeout(activeCopiedTimer);
    activeCopiedTimer = setTimeout(() => {
      activeCopiedBlockIdx = -1;
      const targetBtn = (btn && document.body.contains(btn))
        ? btn
        : document.querySelector(`.code-copy-btn[data-block-idx="${blockIdx}"]`);
      if (targetBtn) {
        targetBtn.textContent = '📋 Copy Code';
        targetBtn.style.color = '#38bdf8';
      }
    }, 1500);
  };

  function restoreActiveCopyState() {
    if (activeCopiedBlockIdx >= 0) {
      const btn = document.querySelector(`.code-copy-btn[data-block-idx="${activeCopiedBlockIdx}"]`);
      if (btn) {
        btn.textContent = '✓ Code Copied!';
        btn.style.color = '#34d399';
      }
    }
  }

  // --- Smart User-Intent-Aware Auto-Scroll Engine ---
  function smartScrollToBottom(force = false) {
    if (!qaScrollBody) return;
    if (force) {
      qaScrollBody.scrollTop = qaScrollBody.scrollHeight;
      return;
    }
    const distanceFromBottom = qaScrollBody.scrollHeight - qaScrollBody.scrollTop - qaScrollBody.clientHeight;
    if (distanceFromBottom < 60) {
      qaScrollBody.scrollTop = qaScrollBody.scrollHeight;
    }
  }

  // --- RAF Token Render Scheduler (60fps batching) ---
  function scheduleAnswerRender(forceScroll = false) {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      if (!answerContent) return;
      const safeText = getRenderSafeAnswer(activeAnswer);
      answerContent.innerHTML = formatMarkdownAnswer(safeText);
      smartScrollToBottom(forceScroll);
      restoreActiveCopyState();
    });
  }

  // --- History Management ---
  function updateHistoryCounter() {
    if (qaHistory.length === 0) {
      if (historyCounter) historyCounter.textContent = '0/0';
      return;
    }
    if (historyCounter) historyCounter.textContent = `${historyIdx + 1}/${qaHistory.length}`;
  }

  function displayHistoryItem(idx) {
    if (idx < 0 || idx >= qaHistory.length) return;
    historyIdx = idx;
    const item = qaHistory[historyIdx];
    if (questionText) questionText.textContent = item.question || '';
    if (answerContent) answerContent.innerHTML = formatMarkdownAnswer(item.answer || '');
    updateHistoryCounter();
    showAnswerCard();
  }

  // --- Live Audio Level Visualizer Engine ---
  let micLevel = 0;
  let sysLevel = 0;
  let targetMicLevel = 0;
  let targetSysLevel = 0;

  window.onAudioLevelUpdate = function (source, levelPct) {
    const val = typeof levelPct === 'number' ? levelPct : parseInt(levelPct, 10) || 0;
    if (source === 'mic') {
      targetMicLevel = Math.min(Math.max(val, 0), 100);
    } else {
      targetSysLevel = Math.min(Math.max(val, 0), 100);
    }
  };

  // Smooth animation frame decay for natural bounce
  function renderAudioVisualizer() {
    // Linear interpolation for smooth spring-like bounce
    micLevel += (targetMicLevel - micLevel) * 0.35;
    sysLevel += (targetSysLevel - sysLevel) * 0.35;
    targetMicLevel *= 0.88; // decay
    targetSysLevel *= 0.88; // decay

    const micBars = document.querySelectorAll('#mic-wave-bars .wave-bar');
    const sysBars = document.querySelectorAll('#system-wave-bars .wave-bar');

    if (micBars.length > 0) {
      micBars.forEach((bar, idx) => {
        const variance = [0.85, 1.25, 1.05, 0.75][idx % 4];
        const h = Math.max(3, Math.min(16, (micLevel * 0.2 * variance)));
        bar.style.height = `${h}px`;
        bar.style.opacity = micLevel > 4 ? '1.0' : '0.45';
      });
      const micPill = document.querySelector('.audio-stream-pill.mic-stream');
      if (micPill) {
        if (micLevel > 10) micPill.classList.add('active-speaking');
        else micPill.classList.remove('active-speaking');
      }
    }

    if (sysBars.length > 0) {
      sysBars.forEach((bar, idx) => {
        const variance = [0.75, 1.15, 1.35, 0.85][idx % 4];
        const h = Math.max(3, Math.min(16, (sysLevel * 0.2 * variance)));
        bar.style.height = `${h}px`;
        bar.style.opacity = sysLevel > 4 ? '1.0' : '0.45';
      });
      const sysPill = document.querySelector('.audio-stream-pill.system-stream');
      if (sysPill) {
        if (sysLevel > 10) sysPill.classList.add('active-speaking');
        else sysPill.classList.remove('active-speaking');
      }
    }

    requestAnimationFrame(renderAudioVisualizer);
  }
  requestAnimationFrame(renderAudioVisualizer);

  // --- Global Window Hooks Called by Python Backend ---
  window.onTranscriptChunk = function (text) {
    if (liveSubtitleText && text) {
      liveSubtitleText.textContent = text;
    }
  };

  window.onQuestionDetected = function (data) {
    currentGenerationId++;
    const qText = typeof data === 'string' ? data : (data.question || '');
    activeQuestion = qText;
    activeAnswer = '';
    renderPending = false;
    if (questionText) questionText.textContent = activeQuestion;
    if (answerContent) answerContent.innerHTML = '<span class="generation-status">⚡ Preparing answer...</span>';
    if (generationStatus) generationStatus.textContent = '⚡ Generating...';
    showAnswerCard();
    smartScrollToBottom(true);
  };

  window.onAnswerStart = function (data) {
    currentGenerationId++;
    activeAnswer = '';
    isStreaming = true;
    renderPending = false;
    if (answerContent) answerContent.innerHTML = '';
    if (generationStatus) generationStatus.textContent = '⚡ Generating...';
    showAnswerCard();
    smartScrollToBottom(true);
  };

  window.onAnswerToken = function (data) {
    const token = typeof data === 'string' ? data : (data.token || '');
    const genId = typeof data === 'object' && data.generation_id !== undefined ? data.generation_id : null;
    if (genId !== null && genId !== currentGenerationId) {
      return; // Stale token discard
    }
    activeAnswer += token;
    scheduleAnswerRender();
  };

  window.onAnswerComplete = function (data) {
    const fullText = typeof data === 'string' ? data : (data.full_text || activeAnswer);
    const genId = typeof data === 'object' && data.generation_id !== undefined ? data.generation_id : null;
    if (genId !== null && genId !== currentGenerationId) {
      return; // Ignore stale completion from previous generation
    }
    activeAnswer = fullText;
    isStreaming = false;
    renderPending = false;
    if (answerContent) {
      answerContent.innerHTML = formatMarkdownAnswer(activeAnswer);
      smartScrollToBottom();
      restoreActiveCopyState();
    }
    if (generationStatus) generationStatus.textContent = '✓ Complete';
    
    // Save clean raw text to history
    if (activeQuestion || activeAnswer) {
      qaHistory.push({
        question: activeQuestion || 'General Discussion',
        answer: activeAnswer,
        timestamp: new Date().toLocaleTimeString()
      });
      historyIdx = qaHistory.length - 1;
      updateHistoryCounter();
    }
  };

  window.onAnswerCancelled = function () {
    currentGenerationId++;
    isStreaming = false;
    renderPending = false;
    if (generationStatus) generationStatus.textContent = '⏸️ Listening...';
  };

  window.onAudioStatusChange = function (status, listening) {
    isTranscribing = listening ?? true;
    if (btnSttToggle) {
      if (isTranscribing) {
        btnSttToggle.classList.add('active');
        if (sttText) sttText.textContent = 'Start';
      } else {
        btnSttToggle.classList.remove('active');
        if (sttText) sttText.textContent = 'Paused';
      }
    }
  };

  window.onTeleprompterStatus = function (active, speed) {
    isTeleprompterActive = active;
    teleprompterSpeed = speed || 1.0;
    if (isTeleprompterActive) {
      startTeleprompterLoop();
    } else {
      stopTeleprompterLoop();
    }
  };

  // --- Teleprompter Animation Loop ---
  function startTeleprompterLoop() {
    stopTeleprompterLoop();
    function step() {
      if (qaScrollBody && isTeleprompterActive) {
        qaScrollBody.scrollTop += (teleprompterSpeed * 0.8);
        teleprompterRafId = requestAnimationFrame(step);
      }
    }
    teleprompterRafId = requestAnimationFrame(step);
  }

  function stopTeleprompterLoop() {
    if (teleprompterRafId) {
      cancelAnimationFrame(teleprompterRafId);
      teleprompterRafId = null;
    }
  }

  // --- User Action Handlers (JS -> Python Bridge) ---
  window.manualGenerateAnswer = function () {
    const text = (manualInputField?.value || '').trim();
    if (text) {
      submitManualQuestion();
    } else {
      showAnswerCard();
      callPython('manual_generate_answer');
    }
  };

  window.triggerAnalyseScreen = function () {
    showAnswerCard();
    activeQuestion = '🖥️ Screen Coding Problem Analysis';
    activeAnswer = '';
    if (questionText) questionText.textContent = activeQuestion;
    if (answerContent) answerContent.innerHTML = '<span class="generation-status">⚡ Capturing screen & analyzing with GPT-4o Vision...</span>';
    if (generationStatus) generationStatus.textContent = '⚡ Analyzing Screen...';
    callPython('trigger_screenshot_solver');
  };

  window.triggerRegionSnip = function () {
    closeAllMenus();
    showAnswerCard();
    activeQuestion = '✂️ Snipped Region Coding Analysis';
    activeAnswer = '';
    if (questionText) questionText.textContent = activeQuestion;
    if (answerContent) answerContent.innerHTML = '<span class="generation-status">✂️ Drag and select a region on your screen to analyze...</span>';
    if (generationStatus) generationStatus.textContent = '✂️ Snipping...';
    callPython('trigger_region_snip');
  };

  window.handleManualInputKeyDown = function (e) {
    if (e.key === 'Enter') {
      submitManualQuestion();
    }
  };

  window.submitManualQuestion = function () {
    const text = (manualInputField?.value || '').trim();
    if (!text) return;
    manualInputField.value = '';
    
    activeQuestion = text;
    if (questionText) questionText.textContent = text;
    showAnswerCard();
    callPython('submit_manual_question', text);
  };

  window.toggleSpeechTranscription = function () {
    callPython('toggle_transcription');
  };

  window.clearTranscript = function () {
    activeQuestion = '';
    activeAnswer = '';
    if (questionText) questionText.textContent = '';
    if (answerContent) answerContent.innerHTML = '';
    if (liveSubtitleText) liveSubtitleText.textContent = 'Listening for speech...';
    dismissAnswerCard();
    callPython('clear_transcript');
  };

  window.refineAnswer = function (mode, prompt) {
    if (generationStatus) generationStatus.textContent = '⚡ Refining...';
    showAnswerCard();
    callPython('refine_answer', mode, prompt);
  };

  window.triggerSystemDesign = function () {
    showAnswerCard();
    callPython('trigger_system_design');
  };

  window.triggerStarStory = function () {
    showAnswerCard();
    callPython('trigger_star_story');
  };

  window.triggerRunCode = function () {
    callPython('trigger_run_code');
  };

  window.triggerTeleprompter = function () {
    closeAllMenus();
    callPython('toggle_teleprompter');
  };

  window.minimizeToBubble = function () {
    const hud = document.getElementById('hud-app');
    const bubble = document.getElementById('floating-app-bubble');
    if (hud) hud.classList.add('hidden');
    if (bubble) bubble.classList.remove('hidden');
    closeAllMenus();
    callPython('set_bubble_mode', true);
  };

  window.restoreFromBubble = function () {
    const hud = document.getElementById('hud-app');
    const bubble = document.getElementById('floating-app-bubble');
    if (bubble) bubble.classList.add('hidden');
    if (hud) hud.classList.remove('hidden');
    callPython('set_bubble_mode', false);
  };

  window.closeApp = function () {
    callPython('close_window');
  };

  window.endSession = function () {
    closeAllMenus();
    callPython('end_session');
    window.location.href = 'home.html';
  };

  window.exportSessionReport = function () {
    closeAllMenus();
    let history = qaHistory && qaHistory.length > 0 ? qaHistory : [];
    if (history.length === 0 && activeQuestion && activeAnswer) {
      history = [{ question: activeQuestion, answer: activeAnswer, timestamp: new Date().toLocaleTimeString() }];
    }
    if (history.length === 0) {
      window.showToast?.('⚠️ No Q&A history to export yet.');
      return;
    }

    const now = new Date();
    const dateStr = now.toISOString().slice(0, 10);
    const timeStr = now.toTimeString().slice(0, 8);
    let md = `# ✦ BluJay AI — Interview Session Report\n\n`;
    md += `**Date:** ${dateStr} ${timeStr}  \n`;
    md += `**Total Questions Practiced:** ${history.length}  \n\n`;
    md += `---\n\n`;

    history.forEach((item, idx) => {
      md += `### Question ${idx + 1} (${item.timestamp || 'Recorded'})\n\n`;
      md += `> **${item.question}**\n\n`;
      md += `#### AI Recommended Spoken Answer:\n\n${item.answer}\n\n`;
      md += `---\n\n`;
    });

    const filename = `BluJay_Interview_Report_${dateStr}_${now.getHours()}${now.getMinutes()}.md`;
    callPython('export_session_report', md, filename);
    window.showToast?.('📄 Exporting session report...');
  };

  window.copyAnswer = function () {
    if (activeAnswer) {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(activeAnswer).catch(() => {});
      }
      callPython('copy_to_clipboard', activeAnswer);
      const btn = document.getElementById('btn-copy-ans');
      if (btn) {
        btn.textContent = '✓ Copied';
        setTimeout(() => { btn.textContent = '📋 Copy'; }, 1500);
      }
    }
  };

  window.navHistory = function (delta) {
    if (qaHistory.length === 0) return;
    let nextIdx = historyIdx + delta;
    if (nextIdx >= 0 && nextIdx < qaHistory.length) {
      displayHistoryItem(nextIdx);
    }
  };

  // --- Menu & Modals Management ---
  function areAllModalsClosed() {
    return document.querySelectorAll('.modal-backdrop:not(.hidden)').length === 0;
  }

  window.toggleMoreMenu = function (e) {
    if (e) e.stopPropagation();
    if (!morePopupMenu) return;

    const willOpen = morePopupMenu.classList.contains('hidden');
    if (willOpen) {
      const btn = document.getElementById('btn-more-menu');
      if (btn) {
        const rect = btn.getBoundingClientRect();
        morePopupMenu.style.top = `${Math.round(rect.bottom + 6)}px`;
        morePopupMenu.style.left = `${Math.max(10, Math.round(rect.left - 20))}px`;
      }
      morePopupMenu.classList.remove('hidden');
      if (!isCardOpen) {
        callPython('resize_window', true);
      }
    } else {
      morePopupMenu.classList.add('hidden');
      if (!isCardOpen && areAllModalsClosed()) {
        callPython('resize_window', false);
      }
    }
  };

  window.toggleDrawer = function (e) {
    if (e) e.stopPropagation();
    openModal('modal-settings');
  };

  window.triggerMockInterview = function () {
    closeAllMenus();
    callPython('trigger_mock_interview');
  };

  function closeAllMenus() {
    if (morePopupMenu && !morePopupMenu.classList.contains('hidden')) {
      morePopupMenu.classList.add('hidden');
      if (!isCardOpen && areAllModalsClosed()) {
        callPython('resize_window', false);
      }
    }
  }

  document.addEventListener('click', function (e) {
    if (morePopupMenu && !morePopupMenu.contains(e.target) && e.target.id !== 'btn-more-menu') {
      const wasOpen = !morePopupMenu.classList.contains('hidden');
      morePopupMenu.classList.add('hidden');
      if (wasOpen && !isCardOpen && areAllModalsClosed()) {
        callPython('resize_window', false);
      }
    }
  });

  window.openModal = function (modalId) {
    closeAllMenus();
    const m = document.getElementById(modalId);
    if (m) {
      m.classList.remove('hidden');
      if (!isCardOpen) {
        callPython('resize_window', true);
      }
    }
  };

  window.closeModal = function (modalId) {
    const m = document.getElementById(modalId);
    if (m) m.classList.add('hidden');
    if (!isCardOpen && areAllModalsClosed() && (!morePopupMenu || morePopupMenu.classList.contains('hidden'))) {
      callPython('resize_window', false);
    }
  };

  window.handleBackdropClick = function (e, modalId) {
    if (e.target.id === modalId) {
      closeModal(modalId);
    }
  };

  window.saveSettings = function () {
    const company = document.getElementById('settings-company')?.value || '';
    const jd = document.getElementById('settings-jd')?.value || '';
    const instructions = document.getElementById('settings-instructions')?.value || '';
    const model = document.getElementById('settings-model')?.value || 'gpt-4o-mini';
    const audioSource = document.getElementById('settings-audio-source')?.value || 'dual';
    
    callPython('update_settings', { company, job_description: jd, instructions, model });
    callPython('update_audio_source', audioSource);
    closeModal('modal-settings');
  };

  // --- HUD Floating Toast Notification ---
  window.showToast = function (message) {
    let toast = document.getElementById('hud-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'hud-toast';
      toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(15, 23, 42, 0.92);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
        z-index: 99999;
        pointer-events: none;
        transition: opacity 0.3s ease, transform 0.3s ease;
        opacity: 0;
      `;
      document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.style.opacity = '1';
    toast.style.transform = 'translateX(-50%) translateY(0)';
    clearTimeout(toast._timeout);
    toast._timeout = setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(-50%) translateY(8px)';
    }, 2400);
  };

  // --- Global Keyboard Shortcuts ---
  window.addEventListener('keydown', function (e) {
    if (e.ctrlKey && e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      manualGenerateAnswer();
    } else if (e.ctrlKey && e.shiftKey && e.key === 'Enter') {
      e.preventDefault();
      triggerAnalyseScreen();
    } else if (e.ctrlKey && e.key === 'Backspace') {
      e.preventDefault();
      clearTranscript();
    } else if (e.altKey && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      copyAnswer();
    } else if (e.altKey && (e.key === 't' || e.key === 'T')) {
      e.preventDefault();
      triggerTeleprompter();
    } else if (e.altKey && (e.key === 'g' || e.key === 'G')) {
      e.preventDefault();
      callPython('toggle_ghost_mode');
    } else if (e.altKey && (e.key === 'c' || e.key === 'C')) {
      e.preventDefault();
      callPython('dock_to_camera');
    } else if (e.ctrlKey && e.key === 'ArrowLeft') {
      e.preventDefault();
      navHistory(-1);
    } else if (e.ctrlKey && e.key === 'ArrowRight') {
      e.preventDefault();
      navHistory(1);
    }
  });

})();
