chrome.contextMenus.create({
    id: "analyzeNews",
    title: "Check Fake News",
    contexts: ["selection"]
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "analyzeNews") {
        chrome.tabs.sendMessage(tab.id, { action: "analyze_news" }, (response) => {
            if (response && response.text) {
                fetch("http://127.0.0.1:8000/verify_news/", {  // Django Backend API
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ news_title: response.text })
                })
                .then(res => res.json())
                .then(data => {
                    chrome.storage.local.set({ analysisResult: data });
                    chrome.runtime.sendMessage({ action: "showResult" });
                })
                .catch(error => console.error("Error:", error));
            }
        });
    }
});
