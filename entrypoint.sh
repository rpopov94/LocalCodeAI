#!/bin/sh

download_model () {
  local model=${1:-$OLLAMA_MODEL}
  if [ -z "$model" ]; then
    echo "Ошибка: переменная OLLAMA_MODEL не задана"
    exit 1
  fi
  if ! ollama list | grep -q "$model"; then
    echo "Скачивается модель: $model"
    ollama pull "$model" || {
      echo "Произошла неожиданная ошибка при скачивании $model"
    }
  else
    echo "Модель $model уже скачана"
  fi
}

ollama serve &

echo "Waiting for Ollama to start..."
while ! curl -s http://localhost:11434 >/dev/null; do
    sleep 1
done


download_model "$OLLAMA_MODEL"

echo "Ollama is ready"
wait