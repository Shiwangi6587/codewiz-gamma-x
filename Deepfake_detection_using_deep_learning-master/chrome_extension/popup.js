document.addEventListener("DOMContentLoaded", function () {
    const checkButton = document.getElementById("checkNews");
    const resultDiv = document.getElementById("result");
    const selectedTextArea = document.getElementById("selectedText");

    // Get the selected text from the active tab
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        if (tabs.length === 0) {
            selectedTextArea.value = "No active tab detected.";
            return;
        }

        chrome.scripting.executeScript(
            {
                target: { tabId: tabs[0].id },
                func: () => window.getSelection().toString(),
            },
            (results) => {
                if (chrome.runtime.lastError || !results || results.length === 0 || !results[0].result) {
                    selectedTextArea.value = "No text selected.";
                } else {
                    selectedTextArea.value = results[0].result;
                }
            }
        );
    });

    // Check the news with the Django backend
    checkButton.addEventListener("click", async () => {
        const newsText = selectedTextArea.value.trim();

        if (newsText === "" || newsText === "No text selected.") {
            resultDiv.innerHTML = "<p style='color:red;'>Please select some text!</p>";
            return;
        }

        try {
            // Update the URL to point to the Django backend
            const response = await fetch("http://127.0.0.1:8000/verify_news/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ news_title: newsText }),
            });

            const data = await response.json();
            resultDiv.innerHTML = ""; // Clear previous results

            // Display fact-check results
            if (data.fact_check && data.fact_check.claims && data.fact_check.claims.length > 0) {
                resultDiv.innerHTML += `<h4>Fact-Check Results:</h4>`;
                data.fact_check.claims.forEach((claim) => {
                    resultDiv.innerHTML += `
                        <p><strong>Claim:</strong> ${claim.text || "Not Available"}<br>
                        <strong>Claimed By:</strong> ${claim.claimant || "Unknown"}<br>
                        <strong>Fact Check:</strong> ${claim.claimReview?.[0]?.textualRating || "Not Rated"}<br>
                        <a href="${claim.claimReview?.[0]?.url || "#"}" target="_blank">Read Full Review</a></p>`;
                });
            } else {
                resultDiv.innerHTML += `<p>No fact-check results found.</p>`;
            }

            // Display related news articles
            if (data.related_news && data.related_news.articles && data.related_news.articles.length > 0) {
                resultDiv.innerHTML += `<h4>Related Articles:</h4>`;
                data.related_news.articles.forEach((article) => {
                    resultDiv.innerHTML += `
                        <p><strong>${article.title || "No Title"}</strong><br>
                        ${article.description || "No Description"}<br>
                        <a href="${article.url || "#"}" target="_blank">Read More</a></p>`;
                });
            } else {
                resultDiv.innerHTML += `<p>No related news articles found.</p>`;
            }
        } catch (error) {
            console.error("Error connecting to the backend:", error);
            resultDiv.innerHTML = `<p style='color:red;'>Error: Unable to connect to the backend.</p>`;
        }
    });
});
