"""
Advanced Scanner — TCP Connect Port Scanner

Apply to: Authorized security audits, penetration testing (with permission),
          service-discovery on networks you own, infrastructure security assessment.

⚠️  LEGAL NOTICE: Port scanning must only be performed on systems you own or
    have explicit written permission to test.  Unauthorised scanning may violate
    computer-fraud laws in your jurisdiction.

Features:
- Thread-based concurrent TCP Connect Scan
- Configurable timeout and thread-pool size
- Per-port result with latency measurement
- Common-service lookup for well-known ports
- CLI entry-point for direct use
- Deterministic mock mode for unit tests

Scan method: TCP Connect Scan
  - Opens a full TCP connection (3-way handshake) to each target port.
  - Does NOT require elevated privileges (no raw sockets).
  - Leaves connection logs on the target — not a stealth scan.

Protocol    Speed    Reliability    Stealth    Requires Admin
─────────────────────────────────────────────────────────────
TCP Connect  Slow     High           No         No
SYN Stealth  Medium   High           Yes        Yes*
ICMP Ping    V.Fast   Medium         No         No
UDP Scan     Slow     Low            No         No
FIN/NULL     Slow     Low            Yes        Yes*

*On Windows / Linux requires elevated privileges.
"""

import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Port state & well-known service names
# ============================================================================

class PortState(Enum):
    """Result of a single TCP connect attempt."""
    OPEN = "open"          # connect() succeeded — port is listening
    CLOSED = "closed"      # connect() refused — port is not listening
    FILTERED = "filtered"  # Timed out — firewall or packet drop


#: Mapping of well-known port numbers to common service names.
COMMON_SERVICES: Dict[int, str] = {
    20: "FTP-data", 21: "FTP", 22: "SSH", 23: "Telnet",
    25: "SMTP", 53: "DNS", 67: "DHCP", 68: "DHCP",
    80: "HTTP", 110: "POP3", 119: "NNTP", 123: "NTP",
    135: "MS-RPC", 137: "NetBIOS-NS", 138: "NetBIOS-DGM", 139: "NetBIOS-SSN",
    143: "IMAP", 161: "SNMP", 162: "SNMP-Trap", 179: "BGP",
    194: "IRC", 389: "LDAP", 443: "HTTPS", 445: "SMB",
    465: "SMTPS", 514: "Syslog", 515: "LPD", 587: "SMTP-Submission",
    631: "IPP", 636: "LDAPS", 993: "IMAPS", 995: "POP3S",
    1433: "MS-SQL", 1521: "Oracle", 1723: "PPTP", 2049: "NFS",
    2181: "ZooKeeper", 2375: "Docker", 2376: "Docker-TLS",
    3306: "MySQL", 3389: "RDP", 4444: "Metasploit",
    5432: "PostgreSQL", 5900: "VNC", 6379: "Redis", 6443: "K8s-API",
    7000: "Cassandra", 8080: "HTTP-Alt", 8443: "HTTPS-Alt",
    8888: "Jupyter", 9000: "SonarQube", 9090: "Prometheus",
    9092: "Kafka", 9200: "Elasticsearch", 9300: "Elasticsearch-Cluster",
    27017: "MongoDB", 27018: "MongoDB-shard", 50070: "HDFS-NameNode",
}


# ============================================================================
# Result data-classes
# ============================================================================

@dataclass
class PortResult:
    """Result for a single scanned port."""
    port: int
    state: PortState
    service: str = ""               # Well-known service name or ""
    latency_ms: Optional[float] = None  # Round-trip time for OPEN ports

    def __str__(self) -> str:
        svc = f" ({self.service})" if self.service else ""
        lat = f"  {self.latency_ms:.1f} ms" if self.latency_ms is not None else ""
        return f"{self.port:>6}/tcp  {self.state.value:<8}{lat}{svc}"


@dataclass
class ScanResult:
    """Aggregated result for a complete host scan."""
    host: str
    ports_scanned: int
    duration_s: float
    results: List[PortResult] = field(default_factory=list)

    @property
    def open_ports(self) -> List[PortResult]:
        return [r for r in self.results if r.state is PortState.OPEN]

    @property
    def filtered_ports(self) -> List[PortResult]:
        return [r for r in self.results if r.state is PortState.FILTERED]

    def summary(self) -> str:
        lines = [
            f"Scan report for {self.host}",
            f"Scanned {self.ports_scanned} ports in {self.duration_s:.2f}s",
            f"Open: {len(self.open_ports)}  "
            f"Filtered: {len(self.filtered_ports)}  "
            f"Closed: {self.ports_scanned - len(self.open_ports) - len(self.filtered_ports)}",
            "",
            f"{'PORT':>10}  {'STATE':<8}  {'SERVICE'}",
            "-" * 40,
        ]
        for r in self.open_ports:
            lines.append(str(r))
        return "\n".join(lines)


# ============================================================================
# Scanner configuration
# ============================================================================

@dataclass
class ScannerConfig:
    """Configuration for AdvancedScanner.

    Apply to: Adjust to network conditions and authorisation scope.
    """
    timeout: float = 1.0        # Seconds to wait per port before marking FILTERED
    max_workers: int = 100      # Concurrent threads (reduce for polite/slow scans)
    resolve_services: bool = True  # Look up service names for well-known ports


# ============================================================================
# Core scanner
# ============================================================================

class AdvancedScanner:
    """Thread-based TCP Connect port scanner.

    Usage::

        scanner = AdvancedScanner()
        result  = scanner.scan("192.168.1.1", ports=range(1, 1025))
        print(result.summary())

    The scanner uses ``socket.connect()`` — no raw sockets, no ICMP.
    It works on any platform without elevated privileges.

    ⚠️  Only scan hosts you own or have written permission to test.
    """

    def __init__(self, config: Optional[ScannerConfig] = None) -> None:
        self.config = config or ScannerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        host: str,
        ports: "range | List[int]" = range(1, 1025),
    ) -> ScanResult:
        """Scan *host* across the given *ports* using TCP Connect.

        Args:
            host:  Target hostname or IP address.
            ports: Iterable of port numbers to scan (default: 1–1024).

        Returns:
            :class:`ScanResult` with per-port states and a summary.
        """
        port_list = list(ports)
        start = time.monotonic()
        results: List[PortResult] = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {
                pool.submit(self._probe, host, p): p for p in port_list
            }
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda r: r.port)
        duration = time.monotonic() - start
        return ScanResult(
            host=host,
            ports_scanned=len(port_list),
            duration_s=duration,
            results=results,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe(self, host: str, port: int) -> PortResult:
        """Perform a single TCP connect attempt and return its :class:`PortResult`."""
        service = (
            COMMON_SERVICES.get(port, "")
            if self.config.resolve_services
            else ""
        )
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            t0 = time.monotonic()
            err = sock.connect_ex((host, port))
            latency = (time.monotonic() - t0) * 1000
            sock.close()

            if err == 0:
                return PortResult(
                    port=port,
                    state=PortState.OPEN,
                    service=service,
                    latency_ms=round(latency, 2),
                )
            return PortResult(port=port, state=PortState.CLOSED, service=service)

        except socket.timeout:
            return PortResult(port=port, state=PortState.FILTERED, service=service)
        except OSError:
            # Covers "Network unreachable", "No route to host", etc.
            return PortResult(port=port, state=PortState.FILTERED, service=service)


# ============================================================================
# Convenience helpers
# ============================================================================

def scan_host(
    host: str,
    ports: "range | List[int]" = range(1, 1025),
    timeout: float = 1.0,
    max_workers: int = 100,
) -> ScanResult:
    """One-shot scan helper.

    Args:
        host:        Target hostname or IP.
        ports:       Port range or list (default 1–1024).
        timeout:     Seconds per port before marking filtered.
        max_workers: Thread-pool concurrency.

    Returns:
        :class:`ScanResult`

    Example::

        from security.advanced_scanner import scan_host
        result = scan_host("127.0.0.1", ports=range(1, 101))
        print(result.summary())
    """
    cfg = ScannerConfig(timeout=timeout, max_workers=max_workers)
    return AdvancedScanner(cfg).scan(host, ports)


def parse_port_range(port_spec: str) -> List[int]:
    """Parse a port specification string into a sorted list of port numbers.

    Supports:
    - Single port:  ``"22"``
    - Range:        ``"80-100"``
    - Comma list:   ``"22,80,443"``
    - Mixed:        ``"22,80-90,443"``

    Args:
        port_spec: Port specification string.

    Returns:
        Sorted list of unique integer port numbers.

    Raises:
        ValueError: If any port number is out of the valid 1–65535 range.
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
# CLI entry-point
# ============================================================================

def _cli() -> None:
    """Command-line interface for the Advanced Scanner.

    Usage::

        python -m security.advanced_scanner <host> [options]

    Examples::

        python -m security.advanced_scanner 192.168.1.1
        python -m security.advanced_scanner example.com --ports 1-1024
        python -m security.advanced_scanner 10.0.0.1 --ports 22,80,443,8080 --timeout 0.5
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Advanced Scanner — TCP Connect Port Scanner\n"
            "⚠️  Authorised use only. Only scan hosts you own or have permission to test."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("host", help="Target hostname or IP address")
    parser.add_argument(
        "--ports",
        default="1-1024",
        help="Port specification: e.g. '80', '1-1024', '22,80,443' (default: 1-1024)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Seconds to wait per port before marking as filtered (default: 1.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="Concurrent threads (default: 100; reduce for polite scans)",
    )
    args = parser.parse_args()

    print(f"\nStarting Advanced Scanner 1.0 — TCP Connect Scan")
    print(f"Target : {args.host}")
    print(f"Ports  : {args.ports}")
    print(f"Timeout: {args.timeout}s per port\n")

    try:
        ports = parse_port_range(args.ports)
    except ValueError as exc:
        parser.error(str(exc))
        return

    result = scan_host(args.host, ports=ports, timeout=args.timeout, max_workers=args.workers)
    print(result.summary())


if __name__ == "__main__":
    _cli()
