const apiUrl = "http://127.0.0.1:8092";

async function sendChat() {
    const question = document.getElementById("question").value;
    const responseElem = document.getElementById("response");
    const chatButton = document.getElementById("chatButton");
    const chatLoading = document.getElementById("chatLoading");

    responseElem.textContent = "";
    chatLoading.style.display = "inline";
    chatButton.disabled = true;

    try {
        const res = await fetch(`${apiUrl}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });

        const data = await res.json();
        responseElem.textContent = data.response;
    } catch (error) {
        responseElem.textContent = "Error: Unable to fetch response.";
    } finally {
        chatLoading.style.display = "none";
        chatButton.disabled = false;
    }
}

async function addKnowledge(event) {
    event.preventDefault();
    const content = document.getElementById("newContent").value;
    const statusElem = document.getElementById("status");
    const addButton = document.getElementById("addButton");
    const addLoading = document.getElementById("addLoading");

    statusElem.textContent = "";
    addLoading.style.display = "inline";
    addButton.disabled = true;

    try {
        const res = await fetch(`${apiUrl}/add_knowledge`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content })
        });

        const data = await res.json();
        statusElem.textContent = data.message;
    } catch (error) {
        statusElem.textContent = "Error: Unable to add knowledge.";
    } finally {
        addLoading.style.display = "none";
        addButton.disabled = false;
    }
}