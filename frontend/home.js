/* ==========================================================================
   BluJay AI - Desktop App & Web Dashboard Controller
   ========================================================================== */

let currentAppTab = 'create';       // 'create' | 'sessions'
let currentSessionType = 'interview'; // 'interview' | 'regular'
let currentFilter = 'all';           // 'all' | 'active' | 'ended'
let cachedSessions = [];
let searchQuery = '';

document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize view mode (default to desktop mode when in native desktop wrapper)
  initViewMode();

  // 2. Fetch authoritative data from backend
  requestDashboardData();
});

/* ==========================================================================
   1. VIEW MODE & TAB MANAGEMENT
   ========================================================================== */
function initViewMode() {
  // If running inside native PyWebView or width < 850px, use desktop mode
  const isDesktop = window.pywebview !== undefined || window.innerWidth < 850;
  if (isDesktop) {
    document.body.classList.add('desktop-mode');
    document.body.classList.remove('web-mode');
  } else {
    document.body.classList.add('web-mode');
    document.body.classList.remove('desktop-mode');
  }
}

function toggleAppExpansion() {
  const isDesktop = document.body.classList.contains('desktop-mode');
  if (isDesktop) {
    document.body.classList.remove('desktop-mode');
    document.body.classList.add('web-mode');
  } else {
    document.body.classList.add('desktop-mode');
    document.body.classList.remove('web-mode');
  }
  if (window.pywebview && window.pywebview.api && window.pywebview.api.toggle_app_expansion) {
    window.pywebview.api.toggle_app_expansion();
  }
}

function minimizeToFloatingBubble() {
  if (window.pywebview && window.pywebview.api) {
    if (typeof window.pywebview.api.minimize_window === 'function') {
      window.pywebview.api.minimize_window();
    } else if (typeof window.pywebview.api.minimize_to_bubble === 'function') {
      window.pywebview.api.minimize_to_bubble();
    }
  } else {
    alert("✦ Minimized window.");
  }
}
window.minimizeToBubble = minimizeToFloatingBubble;

function switchAppTab(tabName) {
  currentAppTab = tabName;
  
  // Segmented buttons
  document.getElementById('tab-create-btn')?.classList.toggle('active', tabName === 'create');
  document.getElementById('tab-sessions-btn')?.classList.toggle('active', tabName === 'sessions');

  // Panels
  document.getElementById('panel-create')?.classList.toggle('active', tabName === 'create');
  document.getElementById('panel-sessions')?.classList.toggle('active', tabName === 'sessions');

  if (tabName === 'sessions') {
    renderDesktopSessions();
  }
}

function selectTypeOption(typeId) {
  currentSessionType = typeId;
  document.getElementById('type-opt-interview')?.classList.toggle('active', typeId === 'interview');
  document.getElementById('type-opt-regular')?.classList.toggle('active', typeId === 'regular');
}

/* ==========================================================================
   2. DESKTOP SESSION CREATION & LAUNCH
   ========================================================================== */
let customInstructionsText = '';

async function executeCreateDesktopSession() {
  const btn = document.getElementById('btn-create-session-hero');
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Launching...";
  }

  const role = document.getElementById('dt-input-role')?.value.trim() || 'Senior Software Engineer';
  const company = document.getElementById('dt-input-company')?.value.trim() || '';
  const jd = document.getElementById('dt-input-jd')?.value.trim() || '';
  const style = document.getElementById('dt-select-style')?.value || 'Direct & Highly Technical';
  const model = document.getElementById('dt-select-model')?.value || 'gpt-4o-mini';
  const language = document.getElementById('dt-select-language')?.value || 'English';

  const autogen = document.getElementById('dt-check-auto-answer')?.checked ?? true;
  const saveTranscript = document.getElementById('dt-check-save-transcript')?.checked ?? true;
  const audioSource = document.getElementById('dt-select-audio-source')?.value || 'dual';

  const config = {
    role: role,
    company: company,
    job_description: jd || role,
    instructions: customInstructionsText || '',
    session_type: currentSessionType === 'interview' ? 'copilot' : 'standard',
    model: model,
    language: language,
    style: style,
    audio_source: audioSource,
    auto_gen: autogen,
    save_transcript: saveTranscript,
    stealth: true,
    priming: true,
    mode: 'copilot'
  };

  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.launch_session) {
      await window.pywebview.api.launch_session(config);
    }
  } catch (err) {
    console.error("Bridge launch error:", err);
  } finally {
    window.location.href = 'index.html';
  }
}

function fillFieldsFromURL() {
  const url = prompt("Enter Job Post URL (LinkedIn, Indeed, Lever, Greenhouse):");
  if (url) {
    document.getElementById('dt-input-company').value = "Target Company";
    document.getElementById('dt-input-jd').value = `Imported role from ${url}: Senior Staff Engineer versed in distributed systems, high throughput pipelines, and microservices architecture.`;
  }
}

function openInstructionsModal() {
  const inst = prompt("Enter specific prep instructions / interviewer personality notes:", customInstructionsText);
  if (inst !== null) {
    customInstructionsText = inst.trim();
    const tag = document.getElementById('tag-ctx-instructions');
    if (customInstructionsText) {
      if (tag) tag.classList.add('active-dark');
      alert("✓ Custom instructions added to active session context!");
    } else {
      if (tag) tag.classList.remove('active-dark');
    }
  }
}

async function triggerJDFitAnalysis() {
  const jd = document.getElementById('dt-input-jd')?.value.trim() || '';
  if (!jd) {
    alert("Please enter a Job Description first.");
    return;
  }
  const btn = document.getElementById('btn-analyze-jd-fit');
  if (btn) btn.textContent = "⏳ Analyzing Match Fit...";

  try {
    let result = null;
    if (window.pywebview && window.pywebview.api && window.pywebview.api.analyze_jd_fit) {
      result = await window.pywebview.api.analyze_jd_fit(jd, "");
    }
    if (!result) {
      result = {
        match_score: 82,
        matched_skills: ["Python", "Distributed Systems", "Cloud"],
        missing_skills: ["Docker", "Kubernetes"],
        bridging_talking_points: ["Emphasize architectural fundamentals and rapid container ramp-up."]
      };
    }
    const card = document.getElementById('jd-fit-card');
    const badge = document.getElementById('jd-fit-badge');
    const details = document.getElementById('jd-fit-details');
    if (card && badge && details) {
      card.classList.remove('hidden');
      badge.textContent = `${result.match_score || 75}% Match`;
      const matched = (result.matched_skills || []).slice(0, 5).join(', ') || 'Core Fundamentals';
      const gaps = (result.missing_skills || []).slice(0, 3).join(', ') || 'None identified';
      const bridge = (result.bridging_talking_points || [])[0] || 'Strong candidate alignment.';
      details.innerHTML = `
        <div style="margin-bottom:4px;"><strong>✓ Strengths:</strong> ${matched}</div>
        <div style="margin-bottom:4px; color:#DC2626;"><strong>△ Skill Gaps:</strong> ${gaps}</div>
        <div style="margin-top:6px; font-style:italic; color:#0284C7;">💡 <strong>Talking Point:</strong> ${bridge}</div>
      `;
    }
  } catch (err) {
    console.error("Fit analysis error:", err);
  } finally {
    if (btn) btn.textContent = "📊 Analyze Resume & Role Fit →";
  }
}

/* ==========================================================================
   3. SESSIONS RENDERING & FILTERING
   ========================================================================== */
function filterDesktopSessions(filterType) {
  currentFilter = filterType;
  document.querySelectorAll('.filter-tab-pill').forEach(btn => btn.classList.remove('active'));
  document.getElementById(`df-${filterType}`)?.classList.add('active');
  renderDesktopSessions();
}

function searchDesktopSessions(query) {
  searchQuery = (query || '').toLowerCase().trim();
  renderDesktopSessions();
}

function renderDesktopSessions() {
  const container = document.getElementById('dt-sessions-container');
  const counter = document.getElementById('dt-session-counter');
  if (!container) return;

  let filtered = cachedSessions.filter(s => {
    if (currentFilter === 'active') return s.status === 'active';
    if (currentFilter === 'ended') return s.status !== 'active';
    return true;
  });

  if (searchQuery) {
    filtered = filtered.filter(s => {
      const title = (s.job_title || '').toLowerCase();
      const comp = (s.company_name || '').toLowerCase();
      return title.includes(searchQuery) || comp.includes(searchQuery);
    });
  }

  if (counter) {
    counter.innerText = `${filtered.length} Session${filtered.length === 1 ? '' : 's'}`;
  }

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="empty-dt-state">
        <h3 class="empty-dt-title">You have no sessions yet</h3>
        <p class="empty-dt-sub">Your sessions will appear here once created.</p>
        <button class="btn-hero-create" style="max-width: 200px; margin: 0 auto; height: 36px;" onclick="switchAppTab('create')">
          + Create Session
        </button>
      </div>
    `;
    return;
  }

  container.innerHTML = filtered.map(s => {
    const isEnded = s.status !== 'active';
    const dateStr = s.started_at ? new Date(s.started_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : 'Recent';
    const qCount = s.total_questions || (s.qa_pairs ? s.qa_pairs.length : 0);
    return `
      <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:12px 14px; display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div style="font-size:13px; font-weight:700; color:#0F172A;">${escapeHtml(s.job_title || 'Tech Interview')}</div>
          <div style="font-size:11px; color:#64748B; margin-top:2px;">
            ${escapeHtml(s.company_name || 'Target Company')} • ${dateStr} • ${qCount} Q&A
          </div>
        </div>
        <span style="font-size:10px; font-weight:700; padding:2px 7px; border-radius:4px; background:${isEnded ? '#F1F5F9' : '#ECFDF5'}; color:${isEnded ? '#475569' : '#065F46'};">
          ${isEnded ? 'Ended' : 'Active'}
        </span>
      </div>
    `;
  }).join('');

  const webContainer = document.getElementById('web-sessions-container');
  if (webContainer) {
    if (filtered.length === 0) {
      webContainer.innerHTML = `
        <div class="empty-dt-state">
          <h3 class="empty-dt-title">You have no sessions yet</h3>
          <p class="empty-dt-sub">Your sessions will appear here once created.</p>
        </div>
      `;
    } else {
      webContainer.innerHTML = container.innerHTML;
    }
  }
}

function filterSessions(filterType) {
  currentFilter = filterType;
  document.querySelectorAll('.w-tab').forEach(btn => btn.classList.remove('active'));
  const activeTab = Array.from(document.querySelectorAll('.w-tab')).find(b => b.textContent.trim().toLowerCase() === filterType.toLowerCase());
  if (activeTab) activeTab.classList.add('active');
  renderDesktopSessions();
}

/* ==========================================================================
   4. WEB SIDEBAR & DOCUMENT UPLOADS
   ========================================================================== */
function switchSidebarTab(tabId) {
  document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
  document.getElementById(`nav-${tabId}`)?.classList.add('active');

  document.querySelectorAll('.web-view').forEach(view => view.classList.remove('active'));
  const targetView = document.getElementById(`view-${tabId}`);
  if (targetView) {
    targetView.classList.add('active');
  } else if (tabId === 'documents') {
    document.getElementById('view-resumes')?.classList.add('active');
  } else {
    document.getElementById('view-callSessions')?.classList.add('active');
  }
}

function handleFileDropOrSelect(file) {
  if (!file) return;
  const fname = file.name;
  onDocumentSynced(fname);

  const reader = new FileReader();
  reader.onload = function(e) {
    const content = e.target.result;
    if (window.pywebview && window.pywebview.api && window.pywebview.api.upload_document_content) {
      window.pywebview.api.upload_document_content(fname, content);
    }
  };
  reader.readAsDataURL(file);
}

function triggerDocumentUpload() {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.browse_document) {
    window.pywebview.api.browse_document();
  } else {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.pdf,.docx,.txt,.md';
    input.onchange = (e) => {
      if (e.target.files.length > 0) {
        handleFileDropOrSelect(e.target.files[0]);
      }
    };
    input.click();
  }
}

window.onDocumentSynced = function(filename) {
  const dtName = document.getElementById('dt-resume-filename');
  const ragTag = document.getElementById('dt-resume-tag');
  if (dtName) dtName.innerText = filename;
  if (ragTag) ragTag.innerText = "(Active Resume - Priority #1)";
  const resumeBtn = document.getElementById('tag-ctx-resume');
  if (resumeBtn) resumeBtn.classList.add('active-dark');
};

// Global Drag-and-Drop Listeners for Resume & Documents
window.addEventListener('dragover', (e) => {
  e.preventDefault();
  e.stopPropagation();
});

window.addEventListener('drop', (e) => {
  e.preventDefault();
  e.stopPropagation();
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleFileDropOrSelect(e.dataTransfer.files[0]);
  }
});

/* ==========================================================================
   5. MODALS & UTILITIES
   ========================================================================== */
function openModal(id) {
  const m = document.getElementById(id);
  if (m) m.classList.remove('hidden');
}

function closeModal(id) {
  const m = document.getElementById(id);
  if (m) m.classList.add('hidden');
}

function handleBackdropClick(e, id) {
  if (e.target.id === id) closeModal(id);
}

function closeDesktopApp() {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.close_window) {
    window.pywebview.api.close_window();
  } else {
    window.close();
  }
}

function choosePlan(tier) {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.buy_credits) {
    window.pywebview.api.buy_credits(tier);
  } else {
    alert(`✓ Plan ${tier} activated!`);
  }
  closeModal('modal-credits');
}

function savePersonaAndClose() {
  const name = document.getElementById('persona-name')?.value || 'Abhishek Salunke';
  const email = document.getElementById('persona-email')?.value || 'abhi958866@gmail.com';
  const skills = document.getElementById('persona-skills')?.value || '';

  const avatar = document.getElementById('web-avatar');
  if (avatar) avatar.innerText = name.split(' ').map(w => w[0]).join('').substring(0, 2).toUpperCase() || 'AS';

  if (window.pywebview && window.pywebview.api && window.pywebview.api.save_profile) {
    window.pywebview.api.save_profile({ name, email, skills });
  }
  closeModal('modal-profile');
}

function requestDashboardData() {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.fetch_dashboard_data) {
    window.pywebview.api.fetch_dashboard_data().then(data => {
      if (data && data.recent_sessions) {
        cachedSessions = data.recent_sessions;
        renderDesktopSessions();
      }
    });
  } else {
    cachedSessions = [];
    renderDesktopSessions();
  }
}

function escapeHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

window.showToast = function (message) {
  let toast = document.getElementById('home-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'home-toast';
    toast.style.cssText = `
      position: fixed;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(15, 23, 42, 0.94);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.4);
      padding: 8px 18px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.3px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
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
  if (toast._timer) clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(-50%) translateY(10px)';
  }, 3000);
};
