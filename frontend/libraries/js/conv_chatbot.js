const apiUrl = "http://127.0.0.1:8092";

async function sendChat() {
    const formData = {
        faiss_settings: {
            embedder_model : document.getElementById("embedder-model").value,
            nearest_neighbor: document.getElementById("nearest-neighbor-number").value,
            similarity_threshold: document.getElementById("similarity-threshold").value,
            distance_metric: document.getElementById("distance-metric").value,
        },
        transformer_settings: {
            skip_special_tokens: document.getElementById("skip-special-tokens").value,
            language_model: document.getElementById("language-model").value,
            max_new_tokens: document.getElementById("max-new-tokens").value,
            temperature: document.getElementById("temperature").value,
            top_k: document.getElementById("top-k").value,
            top_p: document.getElementById("top-p").value,
            repetition_penalty: document.getElementById("repetition-penalty").value,
            length_penalty: document.getElementById("length-penalty").value,
            stop_sequence: document.getElementById("stop-sequence").value,
            seed: document.getElementById("seed").value,
            beam_width: document.getElementById("beam-width").value,
        },
        question: document.getElementById("question").value,
    };

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData),
        });

        const data = await response.json();
        document.getElementById("response").textContent = data.response;
    } catch (error) {
        document.getElementById("response").textContent = "Error: Unable to fetch response.";
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
        // Wait for 10 seconds
        await new Promise(resolve => setTimeout(resolve, 10000));

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