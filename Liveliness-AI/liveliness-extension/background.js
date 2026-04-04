chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "scanWithLivelinessAI",
    title: "Scan with Liveliness-AI",
    contexts: ["image", "video"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "scanWithLivelinessAI") {
    const mediaUrl = info.srcUrl;
    
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: injectUI
    });

    analyzeMedia(mediaUrl, tab.id);
  }
});

async function analyzeMedia(url, tabId) {
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    
    const ext = blob.type.includes('video') ? 'mp4' : 'jpg';

    const formData = new FormData();
    formData.append("file", blob, `scanned_media.${ext}`);

    const res = await fetch("http://127.0.0.1:8000/analyze/", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      throw new Error(`Server returned status: ${res.status}`);
    }

    const data = await res.json();
    
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: updateUIReady,
      args: [data]
    });

  } catch (error) {
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: updateUIError,
      args: [error.message]
    });
  }
}

function injectUI() {
  if (document.getElementById("liveliness-ai-overlay")) return;

  const overlay = document.createElement("div");
  overlay.id = "liveliness-ai-overlay";
  Object.assign(overlay.style, {
    position: "fixed",
    bottom: "20px",
    right: "20px",
    width: "320px",
    padding: "15px",
    backgroundColor: "#1e1e1e",
    color: "#ffffff",
    borderRadius: "10px",
    boxShadow: "0 8px 16px rgba(0,0,0,0.5)",
    zIndex: "2147483647",
    fontFamily: "Arial, sans-serif",
    border: "1px solid #444",
    transition: "all 0.3s ease"
  });

  const closeBtn = document.createElement("button");
  closeBtn.innerText = "✕";
  Object.assign(closeBtn.style, {
    position: "absolute",
    top: "10px",
    right: "10px",
    background: "transparent",
    border: "none",
    color: "#aaa",
    cursor: "pointer",
    fontSize: "14px",
    padding: "0"
  });
  closeBtn.onclick = () => overlay.remove();

  const content = document.createElement("div");
  content.id = "liveliness-ai-content";
  content.innerHTML = `
    <div style="display:flex; align-items:center; gap:12px; margin-top:5px;">
      <div style="width:16px;height:16px;border:3px solid #555;border-top-color:#00a8ff;border-radius:50%;animation:spin 1s linear infinite;"></div>
      <span style="font-size:14px;font-weight:bold;color:#f5f6fa;">Scanning with Liveliness-AI...</span>
    </div>
    <style>@keyframes spin { 100% { transform: rotate(360deg); } }</style>
  `;

  overlay.appendChild(closeBtn);
  overlay.appendChild(content);
  document.body.appendChild(overlay);
}

function updateUIReady(data) {
  const content = document.getElementById("liveliness-ai-content");
  if (!content) return;
  
  const score = data.authenticity_score !== undefined ? data.authenticity_score : null;
  const risk = data.risk_classification || "UNKNOWN";
  const flags = data.flags || [];
  
  let color = "#bdc3c7";
  if (risk === "HIGH" || risk.toUpperCase() === "HIGH") color = "#e74c3c";
  else if (risk === "LOW" || risk.toUpperCase() === "LOW") color = "#2ecc71";
  else if (risk === "MEDIUM" || risk.toUpperCase() === "MEDIUM") color = "#f1c40f";

  content.innerHTML = `
    <h3 style="margin: 0 0 12px 0; font-size:16px; color: #fff; border-bottom: 1px solid #444; padding-bottom: 5px;">Analysis Complete</h3>
    <p style="margin: 0 0 6px 0; font-size:14px;"><strong>Authenticity:</strong> ${score !== null ? score + '%' : 'N/A'}</p>
    <p style="margin: 0 0 6px 0; font-size:14px;"><strong>Risk Level:</strong> <span style="color:${color};font-weight:bold; letter-spacing: 0.5px;">${risk}</span></p>
    <p style="margin: 4px 0 0 0; font-size:12px; color:#aaa; font-style: italic;">${flags.join(', ') || 'No flags reported'}</p>
  `;
}

function updateUIError(msg) {
  const content = document.getElementById("liveliness-ai-content");
  if (!content) return;
  
  content.innerHTML = `
    <h3 style="margin: 0 0 8px 0; font-size:16px; color: #e74c3c;">Connection Error</h3>
    <p style="margin: 0; font-size:13px; color:#bbb; line-height: 1.4;">${msg}</p>
    <p style="margin: 6px 0 0 0; font-size:11px; color:#888;">Ensure local backend is running at http://127.0.0.1:8000</p>
  `;
}
