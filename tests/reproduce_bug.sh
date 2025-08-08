#!/bin/bash

# Reproduce the tool calling issue with ik_llama.cpp
# This curl command replicates the exact request sent by the proxy

curl -X POST http://192.168.0.24:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-12345-empty" \
  -d '{
    "model": "Qwen3",
    "messages": [
      {
        "role": "system",
        "content": "You are Claude Code, Anthropic'\''s official CLI for Claude. You are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user."
      },
      {
        "role": "user",
        "content": "show me latest git commit"
      }
    ],
    "max_tokens": 16384,
    "temperature": 1.0,
    "stream": true,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "Bash",
          "description": "Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.",
          "parameters": {
            "type": "object",
            "properties": {
              "command": {
                "type": "string",
                "description": "The command to execute"
              },
              "description": {
                "type": "string", 
                "description": "Clear, concise description of what this command does in 5-10 words."
              }
            },
            "required": ["command"],
            "additionalProperties": false
          }
        }
      }
    ]
  }'

echo -e "\n\n=== EXPECTED BEHAVIOR ==="
echo "The model should call the Bash tool with command 'git log -1' or similar"
echo "Instead, it likely responds with text like: 'I'\''ll show you the latest git commit by running git log with Bash.'"
echo "Then finishes with finish_reason='stop' instead of making the tool call"