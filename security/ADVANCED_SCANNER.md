# Advanced Scanner — Scanning Methods Documentation

**Version:** Advanced Scanner 1.0  
**Languages:** Português 🇧🇷 | English 🇬🇧 | Русский 🇷🇺  
**Last Updated:** 2026-03-04

---

> ⚠️ **Legal Notice / Aviso Legal / Правовое уведомление**  
> Port scanning must only be performed on systems you own or have **explicit written permission** to test.  
> Unauthorised scanning may violate computer-fraud laws in your jurisdiction.

---

## Table of Contents

- [🇧🇷 Português](#-português)
- [🇬🇧 English](#-english)
- [🇷🇺 Русский](#-русский)
- [📚 Additional Resources](#-additional-resources)

---

## 🇧🇷 PORTUGUÊS

### Como o Scanner Funciona

O Advanced Scanner utiliza **TCP Connect Scan** para detectar portas abertas em máquinas alvo.

### 🔍 Método: TCP Connect Scan

```python
sock = socket(AF_INET, SOCK_STREAM)  # TCP (não ICMP)
sock.settimeout(1)
sock.connect((tgtHost, tgtPort))
```

**Especificações Técnicas:**

| Parâmetro | Valor |
|-----------|-------|
| Protocolo | TCP (Transmission Control Protocol) |
| Família   | AF_INET — IPv4 |
| Tipo      | SOCK_STREAM — TCP com streaming |
| Timeout   | 1 segundo por porta (configurável) |

### 📊 Como Detecta Portas

| Estado | O que ocorre | Resultado |
|--------|-------------|-----------|
| **ABERTA** | `connect()` sucede (handshake TCP completo) | Porta marcada como ABERTA ✓ |
| **FECHADA** | `connect()` falha (connection refused) | Porta marcada como FECHADA ✗ |
| **FILTRADA** | Timeout (firewall bloqueia) | Sem resposta |

### Comparação com Outros Métodos

| Método | Protocolo | Velocidade | Confiabilidade | Stealth | Requer Admin |
|--------|-----------|-----------|---------------|---------|-------------|
| **TCP Connect** | TCP | Lento | Alta | Não | **Não** |
| SYN Stealth | TCP | Médio | Alta | Sim | Sim\* |
| ICMP Ping | ICMP | Muito Rápido | Média | Não | Não |
| UDP Scan | UDP | Lento | Baixa | Não | Não |
| FIN/NULL | TCP | Lento | Baixa | Sim | Sim\* |

\*Em Windows/Linux requer privilégios elevados

### ✅ Vantagens do TCP Connect

- Funciona em qualquer máquina (sem privilégios especiais)
- Muito confiável e preciso
- Funciona através de firewalls (na maioria dos casos)
- Compatível com todas as plataformas
- Fácil de implementar

### ❌ Desvantagens do TCP Connect

- Deixa logs no servidor alvo (não é stealth)
- Mais lento (timeout de 1 segundo por porta)
- Gera muito tráfego de rede
- Pode ser detectado por IDS/IPS
- Desempenho reduzido em redes com alta latência

### 🎯 Quando Usar TCP Connect

- Varreduras legais e autorizadas
- Descoberta de serviços em redes corporativas
- Testes de penetração com permissão
- Avaliação de segurança de infraestrutura

---

## 🇬🇧 ENGLISH

### How the Scanner Works

The Advanced Scanner uses **TCP Connect Scan** to detect open ports on target machines.

### 🔍 Method: TCP Connect Scan

```python
sock = socket(AF_INET, SOCK_STREAM)  # TCP (not ICMP)
sock.settimeout(1)
sock.connect((tgtHost, tgtPort))
```

**Technical Specifications:**

| Parameter | Value |
|-----------|-------|
| Protocol  | TCP (Transmission Control Protocol) |
| Family    | AF_INET — IPv4 |
| Type      | SOCK_STREAM — TCP Streaming |
| Timeout   | 1 second per port (configurable) |

### 📊 How It Detects Ports

| State | What Happens | Result |
|-------|-------------|--------|
| **OPEN** | `connect()` succeeds (full TCP handshake) | Port marked as OPEN ✓ |
| **CLOSED** | `connect()` fails (connection refused) | Port marked as CLOSED ✗ |
| **FILTERED** | Timeout (firewall blocks) | No response |

### Comparison with Other Methods

| Method | Protocol | Speed | Reliability | Stealth | Requires Admin |
|--------|----------|-------|------------|---------|---------------|
| **TCP Connect** | TCP | Slow | High | No | **No** |
| SYN Stealth | TCP | Medium | High | Yes | Yes\* |
| ICMP Ping | ICMP | Very Fast | Medium | No | No |
| UDP Scan | UDP | Slow | Low | No | No |
| FIN/NULL | TCP | Slow | Low | Yes | Yes\* |

\*On Windows/Linux requires elevated privileges

### ✅ Advantages of TCP Connect

- Works on any machine (no special privileges needed)
- Very reliable and accurate
- Works through firewalls (in most cases)
- Compatible with all platforms
- Easy to implement

### ❌ Disadvantages of TCP Connect

- Leaves logs on target server (not stealth)
- Slower (1 second timeout per port)
- Generates significant network traffic
- Can be detected by IDS/IPS
- Slow on networks with high latency

### 🎯 When to Use TCP Connect

- Legal and authorized scans
- Service discovery on corporate networks
- Penetration testing with permission
- Security infrastructure assessment

### 🚀 Quick Start

```bash
# Install no extra dependencies (stdlib only)

# Scan top 1024 ports on localhost
python -m security.advanced_scanner 127.0.0.1

# Scan specific ports
python -m security.advanced_scanner 192.168.1.1 --ports 22,80,443,8080

# Scan a range with a shorter timeout
python -m security.advanced_scanner 10.0.0.1 --ports 1-65535 --timeout 0.5 --workers 200
```

```python
# Programmatic use
from security.advanced_scanner import scan_host, parse_port_range

result = scan_host("127.0.0.1", ports=range(1, 101))
print(result.summary())

for port in result.open_ports:
    print(f"Open: {port.port}/tcp ({port.service})  {port.latency_ms} ms")
```

---

## 🇷🇺 РУССКИЙ

### Как работает сканер

Advanced Scanner использует **TCP Connect Scan** для обнаружения открытых портов на целевых машинах.

### 🔍 Метод: TCP Connect Scan

```python
sock = socket(AF_INET, SOCK_STREAM)  # TCP (не ICMP)
sock.settimeout(1)
sock.connect((tgtHost, tgtPort))
```

**Технические характеристики:**

| Параметр | Значение |
|----------|----------|
| Протокол | TCP (Transmission Control Protocol) |
| Семейство | AF_INET — IPv4 |
| Тип | SOCK_STREAM — TCP потоковая передача |
| Таймаут | 1 секунда на порт (настраивается) |

### 📊 Как он обнаруживает порты

| Состояние | Что происходит | Результат |
|-----------|--------------|-----------|
| **ОТКРЫТ** | `connect()` успешен (полный 3-way handshake TCP) | Порт отмечен как ОТКРЫТ ✓ |
| **ЗАКРЫТ** | `connect()` не удается (соединение отклонено) | Порт отмечен как ЗАКРЫТ ✗ |
| **ФИЛЬТРУЕТСЯ** | Таймаут (брандмауэр блокирует) | Нет ответа |

### Сравнение с другими методами

| Метод | Протокол | Скорость | Надежность | Скрытность | Требует Admin |
|-------|---------|----------|----------|-----------|-------------|
| **TCP Connect** | TCP | Медленно | Высокая | Нет | **Нет** |
| SYN Stealth | TCP | Средняя | Высокая | Да | Да\* |
| ICMP Ping | ICMP | Очень быстро | Средняя | Нет | Нет |
| UDP Scan | UDP | Медленно | Низкая | Нет | Нет |
| FIN/NULL | TCP | Медленно | Низкая | Да | Да\* |

\*В Windows/Linux требуются повышенные привилегии

### ✅ Преимущества TCP Connect

- Работает на любой машине (не требует специальных привилегий)
- Очень надежен и точен
- Работает через брандмауэры (в большинстве случаев)
- Совместим со всеми платформами
- Легко реализуется

### ❌ Недостатки TCP Connect

- Оставляет логи на целевом сервере (не скрытен)
- Медленнее (таймаут 1 секунда на порт)
- Генерирует значительный сетевой трафик
- Может быть обнаружен IDS/IPS
- Медленно работает при высокой задержке сети

### 🎯 Когда использовать TCP Connect

- Легальные и авторизованные сканирования
- Обнаружение сервисов в корпоративных сетях
- Тестирование на проникновение с разрешением
- Оценка безопасности инфраструктуры

---

## 📚 Additional Resources

### TCP Handshake (3-Way Handshake)

```
Client                   Server
  │──── SYN ───────────────►│   Step 1: Client sends SYN
  │◄─── SYN-ACK ────────────│   Step 2: Server responds with SYN-ACK
  │──── ACK ───────────────►│   Step 3: Client sends ACK
  │        connection open   │
```

**TCP Connect Scan result mapping:**

| Outcome | Meaning |
|---------|---------|
| ✅ All 3 steps complete | PORT **OPEN** |
| ❌ RST received after SYN | PORT **CLOSED** |
| ⏱️ No response (timeout) | PORT **FILTERED** |

### Common Port Ranges

| Range | Name | Notes |
|-------|------|-------|
| 0–1023 | Well-known ports | Requires admin on Unix |
| 1024–49151 | Registered ports | Application ports |
| 49152–65535 | Dynamic / Private | Ephemeral ports |

### Well-Known Ports Reference

| Port | Service | Port | Service |
|------|---------|------|---------|
| 22 | SSH | 3306 | MySQL |
| 25 | SMTP | 3389 | RDP |
| 80 | HTTP | 5432 | PostgreSQL |
| 443 | HTTPS | 6379 | Redis |
| 445 | SMB | 8080 | HTTP-Alt |
| 1433 | MS-SQL | 27017 | MongoDB |

### Security Implications

**Detection by IDS/IPS:**
TCP Connect Scan generates a complete TCP handshake for each probe,
producing visible entries in connection logs. Intrusion Detection Systems
(IDS) and Next-Generation Firewalls (NGFW) will typically alert on rapid
sequential connection attempts from a single source IP.

**Firewall interaction:**
- **Stateful firewalls** — will drop packets silently → port appears FILTERED
- **Reject rules** — return RST → port appears CLOSED faster than timeout
- **Accept rules** — complete handshake → port appears OPEN with latency data

### Implementation

The reference implementation lives in [`security/advanced_scanner.py`](advanced_scanner.py).
It uses only Python standard-library modules (`socket`, `concurrent.futures`,
`dataclasses`) — no third-party dependencies required.

```
security/
├── advanced_scanner.py   ← Implementation (this module)
└── ADVANCED_SCANNER.md   ← This documentation
```
