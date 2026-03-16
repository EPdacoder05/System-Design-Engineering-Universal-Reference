"""
Advanced Scanner 1.0 — Multi-Method TCP/UDP/ICMP/SYN/FIN Port Scanner

Apply to: Authorized security audits, penetration testing (with permission),
          service-discovery on networks you own, infrastructure security assessment.

⚠️  LEGAL NOTICE: Port scanning must only be performed on systems you own or
    have explicit written permission to test.  Unauthorised scanning may violate
    computer-fraud laws in your jurisdiction.

Features
--------
- Five scan methods: TCP Connect, SYN Stealth, ICMP Ping, UDP, FIN/NULL/Xmas
- Preset port-lists: web, ssh, database, mail, dns, directory, monitoring, ntp, vpn, common, all
- Rich-based colored progress bar and results table
- Interactive menu (requires questionary)
- Banner grabbing with multi-probe strategy
- Multi-language output: --lang en|pt|ru
- CLI flags: -H, -x, -M, -i, -p, -r, --list, --info, --lang
- Graceful privilege auto-fallback (SYN/ICMP/FIN → TCP when not root)

Scan Methods
────────────
Method       Speed    Stealth  Privilege   Notes
─────────────────────────────────────────────────────────────────────────────
tcp          Slow     ❌ Low   None        Full 3-way handshake; default
syn          Medium*  ✅ High  Admin/Root  Half-open; auto-fallback to tcp
icmp         V.Fast   Medium   Admin/Root  Host discovery; one echo per host
udp          Slow     Medium   None        Connectionless; open|filtered hard
fin          Fast     ✅ High  Admin/Root  FIN/NULL/Xmas flags; auto-fallback

*SYN/FIN are sequential (scapy not thread-safe); large port ranges are slower.

Usage
-----
    python security/advanced_scanner.py -H 192.168.1.1 -x web
    python security/advanced_scanner.py -H 192.168.1.1 -p 22,80,443 -M syn
    python security/advanced_scanner.py -H 192.168.1.1 -r 1-1000 --lang pt
    python security/advanced_scanner.py -i                 # interactive mode
    python security/advanced_scanner.py --list
    python security/advanced_scanner.py --info database
"""

from __future__ import annotations

import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


# ============================================================================
# Scan methods
# ============================================================================

class ScanMethod(Enum):
    TCP  = "tcp"   # Full 3-way handshake (no privileges required)
    SYN  = "syn"   # Half-open SYN scan (requires root/admin; fallback to TCP)
    ICMP = "icmp"  # ICMP echo host discovery (requires root/admin)
    UDP  = "udp"   # Connectionless UDP probe (no privileges required)
    FIN  = "fin"   # FIN/NULL/Xmas stealth (requires root/admin; fallback to TCP)


# ============================================================================
# Preset port lists
# ============================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    "web": {
        "desc": "Web servers — HTTP, HTTPS, Node.js, Flask, dev servers",
        "ports": [80, 443, 3000, 5000, 8080, 8443, 9000, 8888],
    },
    "ssh": {
        "desc": "Remote access — SSH, Telnet, RDP, VNC",
        "ports": [22, 23, 3389, 5900],
    },
    "database": {
        "desc": "Database servers — MySQL, PostgreSQL, MSSQL, Oracle, MongoDB, Redis, CouchDB",
        "ports": [3306, 5432, 1433, 1521, 27017, 6379, 5984],
    },
    "mail": {
        "desc": "Email services — SMTP, POP3, IMAP (plain + TLS)",
        "ports": [25, 110, 143, 587, 993, 995],
    },
    "dns": {
        "desc": "DNS servers",
        "ports": [53],
    },
    "directory": {
        "desc": "Directory services — LDAP, LDAPS, Kerberos, SMB, NetBIOS",
        "ports": [389, 636, 88, 445, 139],
    },
    "monitoring": {
        "desc": "Monitoring — SNMP, SNMP-trap, Netdata, BitTorrent tracker",
        "ports": [161, 162, 19999, 51413],
    },
    "ntp": {
        "desc": "NTP time synchronisation",
        "ports": [123],
    },
    "vpn": {
        "desc": "VPN services — IKE/IPsec, OpenVPN, PPTP",
        "ports": [500, 1194, 1723],
    },
    "common": {
        "desc": "Top 20 most commonly used ports",
        "ports": [
            21, 22, 23, 25, 53, 80, 110, 139, 143, 443,
            445, 993, 995, 1723, 3306, 3389, 5900, 8080, 8443, 27017,
        ],
    },
    "all": {
        "desc": "Comprehensive scan — ports 1–10 000",
        "ports": list(range(1, 10001)),
    },
}


# ============================================================================
# Translations
# ============================================================================

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "title": "Advanced Scanner",
        "target": "Target",
        "method": "Method",
        "ports": "Ports",
        "scanning": "Scanning",
        "open": "OPEN",
        "closed": "CLOSED",
        "filtered": "FILTERED",
        "complete": "Scan Complete",
        "found_open": "{n} open port(s) found",
        "state_col": "State",
        "service_col": "Service",
        "banner_col": "Banner",
        "latency_col": "Latency (ms)",
        "fallback": "⚠  Insufficient privileges for {method} — falling back to TCP",
        "no_scapy": "⚠  scapy not installed — falling back to TCP (pip install scapy)",
        "interactive_prompt": "What do you want to do?",
        "choose_preset": "Choose a preset",
        "choose_method": "Choose scan method",
        "enter_host": "Enter target hostname or IP",
        "enter_ports": "Enter ports (e.g. 22,80,443 or 1-1000)",
    },
    "pt": {
        "title": "Scanner Avançado",
        "target": "Alvo",
        "method": "Método",
        "ports": "Portas",
        "scanning": "Escaneando",
        "open": "ABERTA",
        "closed": "FECHADA",
        "filtered": "FILTRADA",
        "complete": "Scan Concluído",
        "found_open": "{n} porta(s) abertas encontradas",
        "state_col": "Estado",
        "service_col": "Serviço",
        "banner_col": "Banner",
        "latency_col": "Latência (ms)",
        "fallback": "⚠  Sem privilégios para {method} — usando TCP",
        "no_scapy": "⚠  scapy não instalado — usando TCP (pip install scapy)",
        "interactive_prompt": "O que você quer fazer?",
        "choose_preset": "Escolha um preset",
        "choose_method": "Escolha o método de scan",
        "enter_host": "Informe o hostname ou IP alvo",
        "enter_ports": "Informe as portas (ex: 22,80,443 ou 1-1000)",
    },
    "ru": {
        "title": "Продвинутый Сканер",
        "target": "Цель",
        "method": "Метод",
        "ports": "Порты",
        "scanning": "Сканирование",
        "open": "ОТКРЫТ",
        "closed": "ЗАКРЫТ",
        "filtered": "ФИЛЬТРУЕТСЯ",
        "complete": "Сканирование завершено",
        "found_open": "Найдено открытых портов: {n}",
        "state_col": "Состояние",
        "service_col": "Сервис",
        "banner_col": "Баннер",
        "latency_col": "Задержка (мс)",
        "fallback": "⚠  Недостаточно привилегий для {method} — используется TCP",
        "no_scapy": "⚠  scapy не установлен — используется TCP (pip install scapy)",
        "interactive_prompt": "Что вы хотите сделать?",
        "choose_preset": "Выберите пресет",
        "choose_method": "Выберите метод сканирования",
        "enter_host": "Введите адрес или имя хоста",
        "enter_ports": "Введите порты (например 22,80,443 или 1-1000)",
    },
}


def _t(key: str, lang: str = "en", **kwargs: Any) -> str:
    """Return translated string for *key* in *lang*."""
    s = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
    return s.format(**kwargs) if kwargs else s


# ============================================================================
# Port state & common service names
# ============================================================================

class PortState(Enum):
    OPEN     = "open"
    CLOSED   = "closed"
    FILTERED = "filtered"


COMMON_SERVICES: Dict[int, str] = {
    20: "FTP-data",   21: "FTP",         22: "SSH",         23: "Telnet",
    25: "SMTP",       53: "DNS",         67: "DHCP",        68: "DHCP",
    80: "HTTP",       88: "Kerberos",   110: "POP3",       119: "NNTP",
   123: "NTP",       135: "MS-RPC",    137: "NetBIOS-NS", 139: "NetBIOS-SSN",
   143: "IMAP",      161: "SNMP",      162: "SNMP-Trap",  179: "BGP",
   389: "LDAP",      443: "HTTPS",     445: "SMB",        465: "SMTPS",
   500: "IKE",       514: "Syslog",    587: "SMTP-Sub",   631: "IPP",
   636: "LDAPS",     993: "IMAPS",     995: "POP3S",     1194: "OpenVPN",
  1433: "MS-SQL",   1521: "Oracle",   1723: "PPTP",      2049: "NFS",
  2181: "ZooKeeper",2375: "Docker",   2376: "Docker-TLS",3000: "Node/Flask",
  3306: "MySQL",    3389: "RDP",      4444: "MSF",       5000: "Flask/Dev",
  5432: "PostgreSQL",5900: "VNC",     5984: "CouchDB",   6379: "Redis",
  6443: "K8s-API",  7000: "Cassandra",8080: "HTTP-Alt",  8443: "HTTPS-Alt",
  8888: "Jupyter",  9000: "SonarQube",9090: "Prometheus",9092: "Kafka",
  9200: "Elasticsearch",9300: "ES-Cluster",
 19999: "Netdata", 27017: "MongoDB", 27018: "MongoDB-shard",
 50070: "HDFS-NameNode",51413: "BitTorrent",
}


# ============================================================================
# Data-classes
# ============================================================================

@dataclass
class PortResult:
    """Result for a single scanned port."""
    port: int
    state: PortState
    service: str = ""
    latency_ms: Optional[float] = None
    banner: Optional[str] = None       # Banner grabbed after OPEN discovery

    def __str__(self) -> str:
        svc = f" ({self.service})" if self.service else ""
        lat = f"  {self.latency_ms:.1f} ms" if self.latency_ms is not None else ""
        bnr = f"  [{self.banner}]" if self.banner else ""
        return f"{self.port:>6}/tcp  {self.state.value:<8}{lat}{svc}{bnr}"


@dataclass
class ScanResult:
    """Aggregated result for a complete host scan."""
    host: str
    ports_scanned: int
    duration_s: float
    method: str = "tcp"
    results: List[PortResult] = field(default_factory=list)

    @property
    def open_ports(self) -> List[PortResult]:
        return [r for r in self.results if r.state is PortState.OPEN]

    @property
    def filtered_ports(self) -> List[PortResult]:
        return [r for r in self.results if r.state is PortState.FILTERED]

    def summary(self, lang: str = "en") -> str:
        """Return a plain-text summary of the scan results."""
        lines = [
            f"{_t('title', lang)} — Scan report for {self.host}",
            f"Method : {self.method.upper()}  |  "
            f"Scanned {self.ports_scanned} ports in {self.duration_s:.2f}s",
            (
                f"{_t('open', lang)}: {len(self.open_ports)}  "
                f"FILTERED: {len(self.filtered_ports)}  "
                f"{_t('closed', lang)}: "
                f"{self.ports_scanned - len(self.open_ports) - len(self.filtered_ports)}"
            ),
            "",
            f"{'PORT':>10}  {'STATE':<10}  {'SERVICE':<16}  {'BANNER'}",
            "-" * 60,
        ]
        for r in self.open_ports:
            lat = f"{r.latency_ms:.1f}ms" if r.latency_ms is not None else ""
            bnr = (r.banner or "")[:40]
            lines.append(
                f"{r.port:>6}/tcp  "
                f"{_t('open', lang):<10}  "
                f"{r.service:<16}  "
                f"{bnr}"
                + (f"  {lat}" if lat else "")
            )
        return "\n".join(lines)


@dataclass
class ScannerConfig:
    """Scanner configuration.

    Apply to: Adjust timeout/workers for network conditions and scan scope.
    """
    timeout: float = 1.0
    max_workers: int = 100
    resolve_services: bool = True
    grab_banners: bool = False   # Set True to attempt banner grabbing on OPEN ports
    method: ScanMethod = ScanMethod.TCP
    lang: str = "en"


# ============================================================================
# Privilege helpers
# ============================================================================

def _has_root() -> bool:
    """Return True if the current process has root/admin privileges."""
    if sys.platform == "win32":
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    return os.geteuid() == 0


def _scapy_available() -> bool:
    """Return True if scapy can be imported."""
    try:
        import importlib
        importlib.import_module("scapy.all")
        return True
    except ImportError:
        return False


# ============================================================================
# Banner grabbing
# ============================================================================

_HTTP_PROBES = [
    b"GET / HTTP/1.0\r\n\r\n",
    b"HEAD / HTTP/1.0\r\n\r\n",
    b"HELLO\r\n",
    b"QUIT\r\n",
]


def grab_banner(host: str, port: int, timeout: float = 5.0) -> Optional[str]:
    """Attempt to grab a service banner from an open TCP port.

    Strategy (mimics nmap behaviour):
    1. Passive receive — wait for server to send first.
    2. If nothing received, probe with common HTTP/text sequences.

    Args:
        host:    Target host.
        port:    Target port (must be open).
        timeout: Per-attempt timeout in seconds (default 5).

    Returns:
        Decoded banner string (first 256 bytes, printable), or *None*.
    """
    def _readable(raw: bytes) -> str:
        return "".join(
            c if 32 <= ord(c) < 127 else "."
            for c in raw.decode("latin-1", errors="replace")
        )[:256].strip()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        # 1 — passive receive
        try:
            data = sock.recv(256)
            if data:
                sock.close()
                return _readable(data)
        except socket.timeout:
            pass

        # 2 — active probes
        for probe in _HTTP_PROBES:
            try:
                sock.sendall(probe)
                data = sock.recv(256)
                if data:
                    sock.close()
                    return _readable(data)
            except (socket.timeout, OSError):
                continue

        sock.close()
    except Exception:
        pass
    return None


# ============================================================================
# Core scanner
# ============================================================================

class AdvancedScanner:
    """Multi-method TCP/UDP/ICMP/SYN/FIN port scanner.

    Usage::

        scanner = AdvancedScanner()
        result = scanner.scan("192.168.1.1", ports=range(1, 1025))
        print(result.summary())

        # SYN scan (auto-falls-back to TCP if not root)
        cfg = ScannerConfig(method=ScanMethod.SYN)
        result = AdvancedScanner(cfg).scan("192.168.1.1", [22, 80, 443])

    ⚠️  Only scan hosts you own or have written permission to test.
    """

    def __init__(self, config: Optional[ScannerConfig] = None) -> None:
        self.config = config or ScannerConfig()

    # ── public API ────────────────────────────────────────────────────────────

    def scan(
        self,
        host: str,
        ports: "Iterable[int]" = range(1, 1025),
        progress_callback: "Optional[Any]" = None,
    ) -> ScanResult:
        """Scan *host* across *ports* using the configured method.

        Args:
            host:              Target hostname or IP.
            ports:             Iterable of port numbers (default: 1–1024).
            progress_callback: Optional callable(completed, total) for UI.

        Returns:
            :class:`ScanResult`
        """
        method = self.config.method
        port_list = list(ports)

        # Privilege / dependency check for raw-socket methods
        if method in (ScanMethod.SYN, ScanMethod.ICMP, ScanMethod.FIN):
            if not _scapy_available():
                print(_t("no_scapy", self.config.lang))
                method = ScanMethod.TCP
            elif not _has_root():
                print(_t("fallback", self.config.lang, method=method.value.upper()))
                method = ScanMethod.TCP

        start = time.monotonic()

        if method == ScanMethod.TCP:
            results = self._scan_parallel(host, port_list, self._probe_tcp, progress_callback)
        elif method == ScanMethod.UDP:
            results = self._scan_parallel(host, port_list, self._probe_udp, progress_callback)
        elif method == ScanMethod.SYN:
            results = self._scan_sequential(host, port_list, self._probe_syn, progress_callback)
        elif method == ScanMethod.ICMP:
            results = self._scan_icmp(host, port_list, progress_callback)
        elif method == ScanMethod.FIN:
            results = self._scan_sequential(host, port_list, self._probe_fin, progress_callback)
        else:
            results = self._scan_parallel(host, port_list, self._probe_tcp, progress_callback)

        # Optionally grab banners for open ports
        if self.config.grab_banners:
            for r in results:
                if r.state is PortState.OPEN:
                    r.banner = grab_banner(host, r.port, self.config.timeout * 5)

        results.sort(key=lambda r: r.port)
        duration = time.monotonic() - start
        return ScanResult(
            host=host,
            ports_scanned=len(port_list),
            duration_s=duration,
            method=method.value,
            results=results,
        )

    # ── parallel scan (TCP, UDP) ──────────────────────────────────────────────

    def _scan_parallel(self, host, port_list, probe_fn, cb) -> List[PortResult]:
        results: List[PortResult] = []
        completed = 0
        total = len(port_list)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {pool.submit(probe_fn, host, p): p for p in port_list}
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if cb:
                    cb(completed, total)
        return results

    # ── sequential scan (SYN, FIN — scapy not thread-safe) ───────────────────

    def _scan_sequential(self, host, port_list, probe_fn, cb) -> List[PortResult]:
        results: List[PortResult] = []
        total = len(port_list)
        for i, port in enumerate(port_list, 1):
            results.append(probe_fn(host, port))
            if cb:
                cb(i, total)
        return results

    # ── ICMP host-discovery scan ──────────────────────────────────────────────

    def _scan_icmp(self, host, port_list, cb) -> List[PortResult]:
        """Send a single ICMP echo; apply result to all listed ports."""
        host_up = self._probe_icmp_host(host)
        state = PortState.OPEN if host_up else PortState.FILTERED
        total = len(port_list)
        results = []
        for i, port in enumerate(port_list, 1):
            service = COMMON_SERVICES.get(port, "") if self.config.resolve_services else ""
            results.append(PortResult(port=port, state=state, service=service))
            if cb:
                cb(i, total)
        return results

    # ── probe: TCP connect ────────────────────────────────────────────────────

    def _probe_tcp(self, host: str, port: int) -> PortResult:
        service = COMMON_SERVICES.get(port, "") if self.config.resolve_services else ""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            t0 = time.monotonic()
            err = sock.connect_ex((host, port))
            latency = (time.monotonic() - t0) * 1000
            sock.close()
            if err == 0:
                return PortResult(port=port, state=PortState.OPEN,
                                  service=service, latency_ms=round(latency, 2))
            return PortResult(port=port, state=PortState.CLOSED, service=service)
        except socket.timeout:
            return PortResult(port=port, state=PortState.FILTERED, service=service)
        except OSError:
            return PortResult(port=port, state=PortState.FILTERED, service=service)

    # Keep backward-compat alias used by scan_host()
    _probe = _probe_tcp

    # ── probe: UDP ────────────────────────────────────────────────────────────

    def _probe_udp(self, host: str, port: int) -> PortResult:
        service = COMMON_SERVICES.get(port, "") if self.config.resolve_services else ""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.config.timeout)
            sock.sendto(b"\x00", (host, port))
            sock.recv(1024)
            sock.close()
            return PortResult(port=port, state=PortState.OPEN, service=service)
        except socket.timeout:
            # No ICMP unreachable = port is open or filtered
            return PortResult(port=port, state=PortState.OPEN, service=service)
        except ConnectionRefusedError:
            # ICMP port unreachable = closed
            return PortResult(port=port, state=PortState.CLOSED, service=service)
        except OSError:
            return PortResult(port=port, state=PortState.FILTERED, service=service)

    # ── probe: SYN stealth (scapy) ────────────────────────────────────────────

    def _probe_syn(self, host: str, port: int) -> PortResult:
        service = COMMON_SERVICES.get(port, "") if self.config.resolve_services else ""
        try:
            from scapy.all import IP, TCP, sr1, conf  # type: ignore
            conf.verb = 0
            pkt = IP(dst=host) / TCP(dport=port, flags="S", window=65535)
            resp = sr1(pkt, timeout=self.config.timeout, verbose=0)
            if resp is None:
                return PortResult(port=port, state=PortState.FILTERED, service=service)
            if resp.haslayer("TCP"):
                flags = resp["TCP"].flags
                if flags & 0x12:   # SYN-ACK → send RST to reset
                    from scapy.all import send  # type: ignore
                    rst = IP(dst=host) / TCP(dport=port, flags="R", seq=resp["TCP"].ack)
                    send(rst, verbose=0)
                    return PortResult(port=port, state=PortState.OPEN, service=service)
                if flags & 0x14:   # RST-ACK → closed
                    return PortResult(port=port, state=PortState.CLOSED, service=service)
            return PortResult(port=port, state=PortState.FILTERED, service=service)
        except Exception:
            # Any scapy/OS error → fall back to TCP probe
            return self._probe_tcp(host, port)

    # ── probe: ICMP host discovery (scapy) ───────────────────────────────────

    def _probe_icmp_host(self, host: str) -> bool:
        """Return True if host responds to ICMP echo."""
        try:
            from scapy.all import IP, ICMP, sr1, conf  # type: ignore
            conf.verb = 0
            pkt = IP(dst=host) / ICMP()
            resp = sr1(pkt, timeout=self.config.timeout, verbose=0)
            return resp is not None
        except Exception:
            return False

    # ── probe: FIN/NULL/Xmas (scapy) ─────────────────────────────────────────

    def _probe_fin(self, host: str, port: int) -> PortResult:
        service = COMMON_SERVICES.get(port, "") if self.config.resolve_services else ""
        try:
            from scapy.all import IP, TCP, sr1, conf  # type: ignore
            conf.verb = 0
            # FIN packet — RFC 793: no response = open|filtered; RST = closed
            pkt = IP(dst=host) / TCP(dport=port, flags="F")
            resp = sr1(pkt, timeout=self.config.timeout, verbose=0)
            if resp is None:
                return PortResult(port=port, state=PortState.OPEN, service=service)
            if resp.haslayer("TCP") and resp["TCP"].flags & 0x14:  # RST-ACK
                return PortResult(port=port, state=PortState.CLOSED, service=service)
            return PortResult(port=port, state=PortState.FILTERED, service=service)
        except Exception:
            return self._probe_tcp(host, port)


# ============================================================================
# Convenience helper (backward-compatible)
# ============================================================================

def scan_host(
    host: str,
    ports: "Iterable[int]" = range(1, 1025),
    timeout: float = 1.0,
    max_workers: int = 100,
    method: ScanMethod = ScanMethod.TCP,
) -> ScanResult:
    """One-shot scan helper.

    Args:
        host:        Target hostname or IP.
        ports:       Port range or list (default 1–1024).
        timeout:     Seconds per port.
        max_workers: Thread-pool concurrency.
        method:      :class:`ScanMethod` (default TCP).

    Returns:
        :class:`ScanResult`

    Example::

        from security.advanced_scanner import scan_host
        result = scan_host("127.0.0.1", ports=range(1, 101))
        print(result.summary())
    """
    cfg = ScannerConfig(timeout=timeout, max_workers=max_workers, method=method)
    return AdvancedScanner(cfg).scan(host, ports)


def parse_port_range(port_spec: str) -> List[int]:
    """Parse a port specification string into a sorted list of port numbers.

    Supports single ports, ranges, comma lists, and mixed forms::

        "22"          → [22]
        "80-100"      → [80, 81, …, 100]
        "22,80,443"   → [22, 80, 443]
        "22,80-90,443"

    Raises:
        ValueError: If any port is outside 1–65535.
    """
    ports: List[int] = []
    for part in port_spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ports.extend(range(int(lo), int(hi) + 1))
        else:
            ports.append(int(part))
    for p in ports:
        if not (1 <= p <= 65535):
            raise ValueError(f"Port {p} out of range 1–65535")
    return sorted(set(ports))


# ============================================================================
# Rich UI helpers
# ============================================================================

def _rich_print_results(result: ScanResult, lang: str = "en") -> None:
    """Print a Rich-formatted results table and summary."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()

        table = Table(
            title=f"[bold]{_t('title', lang)}[/bold] — {result.host}",
            box=box.ROUNDED,
            show_lines=False,
        )
        table.add_column("Port", style="cyan", justify="right")
        table.add_column(_t("state_col", lang), justify="center")
        table.add_column(_t("service_col", lang), style="dim")
        table.add_column(_t("latency_col", lang), justify="right", style="dim")
        table.add_column(_t("banner_col", lang), style="italic", no_wrap=False)

        state_style = {
            PortState.OPEN:     f"[green]{_t('open', lang)}[/green]",
            PortState.CLOSED:   f"[dim]{_t('closed', lang)}[/dim]",
            PortState.FILTERED: f"[yellow]{_t('filtered', lang)}[/yellow]",
        }

        for r in result.results:
            lat = f"{r.latency_ms:.1f}" if r.latency_ms is not None else ""
            table.add_row(
                f"{r.port}/tcp",
                state_style[r.state],
                r.service,
                lat,
                (r.banner or "")[:60],
            )

        console.print()
        console.print(table)
        n_open = len(result.open_ports)
        console.print(
            f"\n[bold green][✓] {result.ports_scanned} ports scanned[/bold green]  "
            f"[green][+] {n_open} {_t('open', lang).upper()}[/green]  "
            f"[dim][-] {result.ports_scanned - n_open} {_t('closed', lang).upper()}"
            f"/{_t('filtered', lang).upper()}[/dim]"
            f"  [dim]({result.duration_s:.2f}s)[/dim]"
        )
    except ImportError:
        print(result.summary(lang))


def _rich_scan_with_progress(
    scanner: AdvancedScanner,
    host: str,
    ports: List[int],
    lang: str = "en",
) -> ScanResult:
    """Run scan with a Rich progress bar, falling back to plain if Rich absent."""
    try:
        from rich.progress import (
            Progress, SpinnerColumn, BarColumn,
            TextColumn, TimeRemainingColumn, TaskProgressColumn,
        )
        from rich.console import Console
        console = Console()
        _completed: List[int] = [0]

        console.print(
            f"\n[bold]{_t('title', lang)}[/bold]  "
            f"[cyan]{_t('target', lang)}:[/cyan] {host}  "
            f"[cyan]{_t('method', lang)}:[/cyan] "
            f"{scanner.config.method.value.upper()}"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]{_t('scanning', lang)}…[/cyan]"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("", total=len(ports))

            def _cb(done: int, total: int) -> None:
                _completed[0] = done
                progress.update(task, completed=done)

            result = scanner.scan(host, ports, progress_callback=_cb)

        return result

    except ImportError:
        print(f"\n[*] {_t('scanning', lang)} {host} ...")
        return scanner.scan(host, ports)


# ============================================================================
# --list / --info helpers
# ============================================================================

def print_preset_list(lang: str = "en") -> None:
    """Print a table of all presets."""
    try:
        from rich.table import Table
        from rich.console import Console
        from rich import box
        console = Console()
        table = Table(title="Available Presets", box=box.SIMPLE_HEAD)
        table.add_column("Preset", style="cyan bold")
        table.add_column("Ports", justify="right")
        table.add_column("Description", style="dim")
        for name, meta in PRESETS.items():
            ports = meta["ports"]
            count = len(ports) if name != "all" else "10 000"
            table.add_row(name, str(count), meta["desc"])
        console.print(table)
    except ImportError:
        for name, meta in PRESETS.items():
            print(f"  {name:<12}  {meta['desc']}")


def print_preset_info(preset: str, lang: str = "en") -> None:
    """Print detailed info for one preset."""
    if preset not in PRESETS:
        print(f"Preset '{preset}' not found. Run --list to see available presets.")
        return
    meta = PRESETS[preset]
    ports = meta["ports"]
    try:
        from rich.console import Console
        console = Console()
        console.print(f"\n[bold cyan]{preset}[/bold cyan] — {meta['desc']}")
        console.print(f"Ports ({len(ports)}): "
                      + ", ".join(
                          f"[cyan]{p}[/cyan]" + (f"/{COMMON_SERVICES[p]}" if p in COMMON_SERVICES else "")
                          for p in ports[:30]
                      )
                      + ("  …" if len(ports) > 30 else ""))
    except ImportError:
        print(f"\n{preset} — {meta['desc']}")
        print("Ports: " + ", ".join(str(p) for p in ports[:30]) + ("  …" if len(ports) > 30 else ""))


# ============================================================================
# Interactive mode
# ============================================================================

def run_interactive(lang: str = "en") -> None:
    """Launch an interactive menu-driven scan session."""
    try:
        import questionary  # type: ignore
    except ImportError:
        print("questionary not installed — run: pip install questionary")
        return

    from rich.console import Console
    console = Console()
    console.print(f"\n[bold green]{_t('title', lang)}[/bold green] — Interactive Mode\n")

    action = questionary.select(
        _t("interactive_prompt", lang),
        choices=[
            "Scan with preset",
            "Scan specific ports (-p)",
            "Scan port range (-r)",
            "List all presets",
            "Exit",
        ],
    ).ask()

    if not action or action == "Exit":
        return

    if action == "List all presets":
        print_preset_list(lang)
        return

    host = questionary.text(_t("enter_host", lang)).ask()
    if not host:
        return

    # Determine ports
    if action == "Scan with preset":
        preset = questionary.select(
            _t("choose_preset", lang),
            choices=list(PRESETS.keys()),
        ).ask()
        if not preset:
            return
        ports = PRESETS[preset]["ports"]
    elif action == "Scan specific ports (-p)":
        raw = questionary.text(_t("enter_ports", lang), default="22,80,443").ask()
        try:
            ports = parse_port_range(raw or "22,80,443")
        except ValueError as exc:
            print(f"Error: {exc}")
            return
    else:  # port range
        raw = questionary.text("Enter range (e.g. 1-1000)", default="1-1024").ask()
        try:
            ports = parse_port_range(raw or "1-1024")
        except ValueError as exc:
            print(f"Error: {exc}")
            return

    # Method selection
    method_choice = questionary.select(
        _t("choose_method", lang),
        choices=[
            "TCP Connect (default — no admin)",
            "SYN Stealth (admin/root)",
            "ICMP Ping (admin/root)",
            "UDP Scan (no admin)",
            "FIN/NULL/Xmas (admin/root)",
        ],
    ).ask()
    method_map = {
        "TCP Connect (default — no admin)": ScanMethod.TCP,
        "SYN Stealth (admin/root)": ScanMethod.SYN,
        "ICMP Ping (admin/root)": ScanMethod.ICMP,
        "UDP Scan (no admin)": ScanMethod.UDP,
        "FIN/NULL/Xmas (admin/root)": ScanMethod.FIN,
    }
    method = method_map.get(method_choice or "", ScanMethod.TCP)

    cfg = ScannerConfig(method=method, lang=lang)
    scanner = AdvancedScanner(cfg)
    result = _rich_scan_with_progress(scanner, host, list(ports), lang)
    _rich_print_results(result, lang)


# ============================================================================
# CLI
# ============================================================================

def _cli() -> None:
    """Command-line entry-point.

    Examples::

        python security/advanced_scanner.py -H 192.168.1.1 -x web
        python security/advanced_scanner.py -H 192.168.1.1 -p 22,80,443 -M syn
        python security/advanced_scanner.py -H 192.168.1.1 -r 1-1000 --lang pt
        python security/advanced_scanner.py -i
        python security/advanced_scanner.py --list
        python security/advanced_scanner.py --info database
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="advancedscanner",
        description=(
            "Advanced Scanner 1.0 — Multi-Method Port Scanner\n"
            "⚠  Authorised use only. Only scan systems you own or have permission to test."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── scanning target ───────────────────────────────────────────────────────
    target = parser.add_argument_group("Scanning target")
    target.add_argument("-H", "--host", help="Target hostname or IP address")

    # ── port selection ────────────────────────────────────────────────────────
    ports_grp = parser.add_argument_group("Port selection (choose one)")
    mx = ports_grp.add_mutually_exclusive_group()
    mx.add_argument("-x", "--preset", choices=list(PRESETS.keys()),
                    help="Use a named preset port list")
    mx.add_argument("-p", "--ports",
                    help="Comma-separated ports / ranges: '22,80,443' or '80-100,443'")
    mx.add_argument("-r", "--range",
                    help="Port range: '1-1000'")

    # ── scan mode ─────────────────────────────────────────────────────────────
    mode = parser.add_argument_group("Scan mode")
    mode.add_argument("-M", "--method",
                      choices=[m.value for m in ScanMethod],
                      default="tcp",
                      help="Scan method (default: tcp)")
    mode.add_argument("-i", "--interactive",
                      action="store_true",
                      help="Launch interactive menu")

    # ── output / info ─────────────────────────────────────────────────────────
    info = parser.add_argument_group("Information")
    info.add_argument("--list", action="store_true",
                      help="List all available presets and exit")
    info.add_argument("--info", metavar="PRESET",
                      help="Show details for a preset and exit")
    info.add_argument("--lang", choices=["en", "pt", "ru"], default="en",
                      help="Output language (default: en)")
    info.add_argument("--banners", action="store_true",
                      help="Attempt banner grabbing on open ports")
    info.add_argument("--timeout", type=float, default=1.0,
                      help="Seconds per port (default: 1.0)")
    info.add_argument("--workers", type=int, default=100,
                      help="Thread pool size for TCP/UDP (default: 100)")

    args = parser.parse_args()
    lang = args.lang

    # ── info-only commands ────────────────────────────────────────────────────
    if args.list:
        print_preset_list(lang)
        return

    if args.info:
        print_preset_info(args.info, lang)
        return

    if args.interactive:
        run_interactive(lang)
        return

    # ── require host for actual scans ─────────────────────────────────────────
    if not args.host:
        parser.error("argument -H/--host is required (or use -i for interactive mode)")

    # ── resolve ports ─────────────────────────────────────────────────────────
    if args.preset:
        ports = PRESETS[args.preset]["ports"]
    elif args.ports:
        try:
            ports = parse_port_range(args.ports)
        except ValueError as exc:
            parser.error(str(exc))
            return
    elif args.range:
        try:
            ports = parse_port_range(args.range)
        except ValueError as exc:
            parser.error(str(exc))
            return
    else:
        ports = PRESETS["common"]["ports"]

    method = ScanMethod(args.method)
    cfg = ScannerConfig(
        timeout=args.timeout,
        max_workers=args.workers,
        method=method,
        grab_banners=args.banners,
        lang=lang,
    )
    scanner = AdvancedScanner(cfg)
    result = _rich_scan_with_progress(scanner, args.host, list(ports), lang)
    _rich_print_results(result, lang)


if __name__ == "__main__":
    _cli()
