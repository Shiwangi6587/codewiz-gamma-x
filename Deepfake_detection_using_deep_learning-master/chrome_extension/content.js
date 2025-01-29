// Function to get the currently selected text
function getSelectedText() {
    return window.getSelection().toString().trim();
}

// Listener to send the selected text to the popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "getSelectedText") {
        const selectedText = getSelectedText();
        sendResponse({ text: selectedText });
    }
});
