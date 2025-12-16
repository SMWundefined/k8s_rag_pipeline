#!/bin/bash

echo "GenAI 101: RAG Workshop Setup"
echo ""
# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo " Ollama not found!"
    echo ""
    echo "Please install Ollama first:"
    echo "  • Visit: https://ollama.ai"
    echo "  • Or run: curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    exit 1
fi

echo "Ollama found"
echo ""

# Pull llama3.2 model
echo "Pulling llama3.2 model..."
ollama pull llama3.2

echo ""
echo "Downloading K8s documentation..."

# Create docs directory
mkdir -p k8s-data/docs

# Download actual K8s documentation (kubectl commands, concepts, etc.)
curl -s -o k8s-data/docs/kubectl-cheatsheet.md \
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/reference/kubectl/cheatsheet.md"

curl -s -o k8s-data/docs/pods.md \
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/pods/_index.md"

curl -s -o k8s-data/docs/deployments.md \
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/controllers/deployment.md"

curl -s -o k8s-data/docs/services.md \
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/services-networking/service.md"

curl -s -o k8s-data/docs/persistent-volumes.md \
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/storage/persistent-volumes.md"

curl -s -o k8s-data/docs/configmaps.md \
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/configuration/configmap.md"

curl -s -o k8s-data/docs/secrets.md \
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/configuration/secret.md"

curl -s -o k8s-data/docs/debugging-pods.md \
    "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/tasks/debug/debug-application/debug-running-pod.md"

echo "K8s documentation downloaded"

echo ""
echo "Setting up Python environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "================"
echo "Setup complete!"
echo "================"
echo ""
