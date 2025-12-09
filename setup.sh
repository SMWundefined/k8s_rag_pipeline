#!/bin/bash

echo "GenAI 101: RAG Workshop Setup"
echo "================================"
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
echo "Setting up Python environment..."

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "="*50
echo "Setup complete!"
echo "="*50
echo ""
