document.addEventListener("DOMContentLoaded", () => {
  // DOM elements
  const taskInput = document.getElementById("task-input");
  const initializeBtn = document.getElementById("initialize");
  const terminateBtn = document.getElementById("terminate");
  const outputDiv = document.getElementById("output");
  const connectionStatus = document.getElementById("connection-status");

  // Generate a unique client ID
  const clientId = "client_" + Math.random().toString(36).substring(2, 15);
  let socket = null;

  // Connect to WebSocket server
  function connectWebSocket() {
    // Close existing socket if any
    if (socket) {
      socket.close();
    }

    // Create new WebSocket connection
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;
    socket = new WebSocket(wsUrl);

    // Socket event handlers
    socket.onopen = () => {
      connectionStatus.textContent = "Connected";
      connectionStatus.classList.remove("disconnected");
      connectionStatus.classList.add("connected");
      initializeBtn.disabled = false;
    };

    socket.onclose = () => {
      connectionStatus.textContent = "Disconnected";
      connectionStatus.classList.remove("connected");
      connectionStatus.classList.add("disconnected");
      initializeBtn.disabled = true;
      terminateBtn.disabled = true;

      // Attempt to reconnect after delay
      setTimeout(connectWebSocket, 3000);
    };

    socket.onerror = (error) => {
      addMessage("error", "WebSocket error: " + error.message);
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      processMessage(data);
    };
  }

  // Process received messages
  function processMessage(data) {
    const type = data.type;
    const content = data.content;

    switch (type) {
      case "status":
        addMessage("status", content);
        break;
      case "thought":
        addMessage("thought", content);
        break;
      case "tool_call":
        let toolCallContent = `Tool: ${data.tool}`;
        if (data.args) {
          toolCallContent += `\nArguments: ${JSON.stringify(
            data.args,
            null,
            2
          )}`;
        }
        addMessage("tool-call", toolCallContent);
        break;
      case "tool_result":
        let resultContent = `Result from: ${data.tool}`;
        if (data.success === false) {
          resultContent += " (Failed)";
        }
        if (data.content) {
          resultContent += `\n${data.content}`;
        }
        if (data.stdout) {
          resultContent += `\nOutput:\n${data.stdout}`;
        }
        if (data.error) {
          resultContent += `\nError:\n${data.error}`;
        }
        addMessage("tool-result", resultContent);
        break;
      case "code":
        addMessage("code", content);
        break;
      case "error":
        addMessage("error", content);
        break;
      case "final_answer":
        addMessage("final-answer", "Final Answer:\n" + content);
        terminateBtn.disabled = true;
        initializeBtn.disabled = false;
        break;
      case "execution_complete":
        addMessage("status", "Execution complete");
        terminateBtn.disabled = true;
        initializeBtn.disabled = false;
        break;
      default:
        addMessage("status", JSON.stringify(data));
    }

    // Scroll to bottom
    outputDiv.scrollTop = outputDiv.scrollHeight;
  }

  // Add message to output
  function addMessage(type, content) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}`;

    if (type === "code") {
      const pre = document.createElement("pre");
      pre.textContent = content;
      messageDiv.appendChild(pre);
    } else {
      messageDiv.textContent = content;
    }

    outputDiv.appendChild(messageDiv);
  }

  // Initialize agent
  initializeBtn.addEventListener("click", () => {
    const task = taskInput.value.trim();
    if (!task) {
      addMessage("error", "Please enter a task");
      return;
    }

    // Clear previous output
    outputDiv.innerHTML = "";
    addMessage("status", `Initializing agent for task: ${task}`);

    // Disable initialize button, enable terminate
    initializeBtn.disabled = true;
    terminateBtn.disabled = false;

    // Send initialize command
    socket.send(
      JSON.stringify({
        command: "initialize",
        task: task,
      })
    );
  });

  // Terminate agent
  terminateBtn.addEventListener("click", () => {
    addMessage("status", "Terminating agent...");

    // Send terminate command
    socket.send(
      JSON.stringify({
        command: "terminate",
      })
    );

    // Update UI
    terminateBtn.disabled = true;
    initializeBtn.disabled = false;
  });

  // Initial connection
  connectWebSocket();
});
