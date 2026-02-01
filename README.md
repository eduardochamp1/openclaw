# 🦀 OpenClaw Agent

Um agente autônomo de IA com suporte multi-provider (Claude/Gemini), busca na web e gerenciamento de arquivos.

## ✨ Features

- **Multi-Provider LLM**: Suporte a Claude (Anthropic) e Gemini (Google)
- **Web Search**: Busca na internet com DuckDuckGo (gratuito), Serper ou Tavily
- **File Manager**: Leitura, escrita e organização de arquivos
- **Memória Persistente**: Histórico de conversas e fatos aprendidos
- **Motor de Raciocínio**: Ciclo cognitivo POAR (Perceive-Orient-Act-Reflect)
- **Docker Ready**: Deploy containerizado

## 🚀 Quick Start

### 1. Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/openclaw.git
cd openclaw

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt
```

### 2. Configuração

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite com suas API keys
nano .env
```

**API Keys necessárias (pelo menos uma):**
- `ANTHROPIC_API_KEY` - Para usar Claude ([console.anthropic.com](https://console.anthropic.com))
- `GOOGLE_API_KEY` - Para usar Gemini ([aistudio.google.com](https://aistudio.google.com))

### 3. Execução

```bash
# Modo interativo
python main.py

# Query única
python main.py "Pesquise as últimas notícias sobre IA"

# Com provider específico
python main.py -p gemini "Qual a previsão do tempo?"

# Verificar configuração
python main.py check
```

## 🐳 Docker

```bash
cd docker

# Build e run
docker-compose up -d

# Modo interativo
docker-compose run openclaw

# Ver logs
docker-compose logs -f
```

## 📁 Estrutura do Projeto

```
openclaw/
├── core/
│   ├── agent.py          # Agent principal
│   ├── memory.py         # Sistema de memória
│   └── reasoning.py      # Motor de raciocínio POAR
├── providers/
│   └── llm_provider.py   # Providers Claude e Gemini
├── tools/
│   ├── web_search.py     # Ferramenta de busca
│   └── file_manager.py   # Gerenciamento de arquivos
├── config/
│   └── settings.yaml     # Configurações
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── main.py               # Ponto de entrada
└── requirements.txt
```

## ⚙️ Configuração

Edite `config/settings.yaml`:

```yaml
agent:
  name: "OpenClaw"
  max_iterations: 15
  thinking_enabled: true

providers:
  default: "claude"  # ou "gemini"
  claude:
    model: "claude-sonnet-4-20250514"
    max_tokens: 8192
  gemini:
    model: "gemini-2.0-flash"
    max_tokens: 8192

tools:
  web_search:
    enabled: true
    provider: "duckduckgo"  # gratuito, sem API key
  file_manager:
    enabled: true
    workspace: "./workspace"
```

## 🔧 Comandos no Modo Interativo

| Comando | Descrição |
|---------|-----------|
| `exit` / `quit` | Encerrar |
| `clear` | Limpar conversa |
| `memory` | Ver resumo da memória |

## 📚 Uso como Biblioteca

```python
import asyncio
from openclaw import OpenClawAgent, AgentConfig

async def main():
    # Criar agent
    agent = OpenClawAgent()
    
    # Executar query
    response = await agent.run("Pesquise sobre Python 3.12")
    print(response)
    
    # Ou modo interativo
    await agent.interactive()

asyncio.run(main())
```

## 🛠️ Adicionando Novas Tools

1. Crie um arquivo em `tools/`:

```python
# tools/my_tool.py
class MyTool:
    @property
    def definition(self) -> dict:
        return {
            "name": "my_tool",
            "description": "O que a tool faz",
            "parameters": {...}
        }
    
    async def execute(self, **kwargs) -> dict:
        # Implementação
        return {"success": True, "result": "..."}
```

2. Registre em `core/agent.py`

## 🔒 Segurança

- Arquivos são restritos ao workspace
- Extensões de arquivo são validadas
- Sem acesso a paths fora do diretório permitido

## 📝 Licença

MIT License

## 🤝 Contribuição

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Add nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

**Feito com 🦀 por OpenClaw Team**
