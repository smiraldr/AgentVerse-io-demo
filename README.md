# 🤖 AgentVerse IO Intelligence Demo 🪐

This project is a **fork** of the [OpenBMB/AgentVerse](https://github.com/OpenBMB/AgentVerse) framework. It has been customized to work with **io intelligence** and packaged as a ready-to-run Docker demo. In this workshop, you will run a simulation of a **council discussion** where multiple AI agents deliberate on **DePIN (Decentralized Physical Infrastructure Networks)**.

---

## ⚙️ Prerequisites

### 1. Install Docker

Follow the guide for your operating system:

- **Ubuntu:** [Install Docker on Ubuntu](https://docs.io.net/docs/ubuntu-install-docker)
- **Windows:** [Install Docker on Windows](https://docs.io.net/docs/install-docker-on-windows)
- **macOS:** [Install Docker on macOS](https://docs.io.net/docs/install-docker-on-macos)

### 2. Generate io.net API Key

Sign up at [https://io.net](https://io.net) and obtain your API key. Enjoy io intelligence capabilities for free!

### 3. Pull the Prebuilt Docker Image

```bash
docker pull smiraldr/agentverse-io-intelligence-demo
```

---

## 🚀 Running the Demo

### Step 1: Create the Local Directory and Add Your API Key

Create a local directory named `agentverse-test` and inside it create a file named `.env` with the following content:

```env
VLLM_API_KEY=BRINGYOURKEYFROMIOINTELLIGENCE
VLLM_BASE_URL=https://api.intelligence.io.solutions/api/v1
```

Your folder structure should look like this:

```
agentverse-test/
└── .env
```

### Step 2: Run the Container

Open your terminal, navigate into the `agentverse-test` folder, and then run:

```bash
docker run --env-file ./.env --network host --platform linux/amd64 smiraldr/agentverse-io-intelligence-demo sh -c "agentverse-simulation-gui --task /app/AgentVerse/agentverse/tasks/simulation/depin_council"
```

### Step 3: View the Simulation

Open your browser and go to:

```
http://localhost:7860
```

You’ll see a simulation of AI agents discussing opportunities, challenges, and trends around **DePIN**.

---

## 💡 Additional Examples

Other examples are available in the repository, showcasing a variety of scenarios. **Note:** These additional examples can be more complex as they may integrate tool calls. For these cases, some developer experience and knowledge of LLM capabilities—such as choosing the right type of LLM for the task—is recommended. You can extend the functionality by vibe coding with LLMs using IO Intelligence. IO Intelligence supports OpenAI standards and can be used as a drop-in replacement for any application that works with the OpenAI Developer API.

---

## 🙌 Credits

- Forked from [OpenBMB/AgentVerse](https://github.com/OpenBMB/AgentVerse)
- Powered by [io.net](https://io.net)

Enjoy the demo and have fun exploring multi-agent collaboration!
