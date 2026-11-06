"""In-memory PCAP/PCAPNG packet capture parser implementing BaseFileParser."""

import io
import math
import struct
from typing import Any, Dict, List, Optional
import pandas as pd

from src.interfaces.parser import BaseFileParser
from src.security.sanitizer import sanitize_dataframe_text_vectors


def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy for payload bytes."""
    if not data:
        return 0.0
    entropy = 0.0
    total_len = len(data)
    counts: Dict[int, int] = {}
    for b in data:
        counts[b] = counts.get(b, 0) + 1
    for count in counts.values():
        p_x = count / total_len
        if p_x > 0:
            entropy -= p_x * math.log2(p_x)
    return round(entropy, 3)


class PCAPFileParser(BaseFileParser):
    """In-memory parser for PCAP and PCAPNG network packet captures."""

    def parse(self, buffer: io.BytesIO, filename: str) -> pd.DataFrame:
        """Parse in-memory PCAP byte stream into tabular telemetry DataFrame.

        Extracts network layer (IPv4/IPv6, TTL), transport layer (TCP/UDP ports,
        flags, packet sizes), and payload entropy features.

        Args:
            buffer: In-memory byte stream.
            filename: Source file name.

        Returns:
            Sanitized pd.DataFrame of network packet vectors.

        Raises:
            ValueError: If PCAP is corrupt or contains no readable IP packets.
        """
        buffer.seek(0)
        raw_bytes = buffer.read()
        buffer.seek(0)

        if len(raw_bytes) < 24:
            raise ValueError(f"PCAP file '{filename}' is too short (corrupt header).")

        records: List[Dict[str, Any]] = []

        # 1. Attempt parsing with Scapy if available
        try:
            from scapy.all import IP, IPv6, TCP, UDP, ICMP, Ether
            from scapy.utils import PcapReader

            pcap_reader = PcapReader(io.BytesIO(raw_bytes))
            first_ts: Optional[float] = None

            for idx, pkt in enumerate(pcap_reader):
                if idx >= 5000:  # Cap parsing to 5,000 packets for rapid AutoML execution
                    break

                ts = float(pkt.time)
                if first_ts is None:
                    first_ts = ts
                rel_ts = round(ts - first_ts, 4)

                mac = pkt[Ether].src if pkt.haslayer(Ether) else "00:00:00:00:00:00"

                if pkt.haslayer(IP):
                    ip_layer = pkt[IP]
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst
                    ttl = int(ip_layer.ttl)
                    pkt_len = int(ip_layer.len)
                elif pkt.haslayer(IPv6):
                    ip6_layer = pkt[IPv6]
                    src_ip = ip6_layer.src
                    dst_ip = ip6_layer.dst
                    ttl = int(ip6_layer.hlim)
                    pkt_len = int(len(pkt))
                else:
                    continue  # Non-IP packets ignored

                # Transport layer
                sport = 0
                dport = 0
                proto = "OTHER"
                tcp_flags = "NONE"

                if pkt.haslayer(TCP):
                    tcp_layer = pkt[TCP]
                    sport = int(tcp_layer.sport)
                    dport = int(tcp_layer.dport)
                    proto = "TCP"
                    tcp_flags = str(tcp_layer.flags)
                elif pkt.haslayer(UDP):
                    udp_layer = pkt[UDP]
                    sport = int(udp_layer.sport)
                    dport = int(udp_layer.dport)
                    proto = "UDP"
                elif pkt.haslayer(ICMP):
                    proto = "ICMP"

                payload = bytes(pkt.payload.payload) if hasattr(pkt.payload, "payload") else b""
                payload_len = len(payload)
                entropy = calculate_entropy(payload)

                records.append({
                    "packet_index": idx + 1,
                    "relative_time_sec": rel_ts,
                    "source_ip": src_ip,
                    "destination_ip": dst_ip,
                    "source_port": sport,
                    "destination_port": dport,
                    "protocol": proto,
                    "packet_length": pkt_len,
                    "payload_size": payload_len,
                    "time_to_live": ttl,
                    "tcp_flags": tcp_flags,
                    "shannon_entropy": entropy,
                    "mac_address": mac,
                })

        except Exception:
            # 2. Pure Python Fallback parser for standard libpcap files
            records = self._parse_libpcap_pure_python(raw_bytes)

        if not records:
            raise ValueError(
                f"No IP packets could be decoded from PCAP '{filename}'. "
                "Ensure the capture contains Ethernet IP traffic."
            )

        df = pd.DataFrame(records)
        return sanitize_dataframe_text_vectors(df)

    def _parse_libpcap_pure_python(self, data: bytes) -> List[Dict[str, Any]]:
        """Fallback raw struct parser for standard .pcap binary format."""
        if len(data) < 24:
            return []

        magic = data[:4]
        if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
            endian = "<"
        elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
            endian = ">"
        else:
            return []

        offset = 24
        records: List[Dict[str, Any]] = []
        first_sec: Optional[int] = None
        pkt_idx = 0

        while offset + 16 <= len(data) and pkt_idx < 5000:
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(
                f"{endian}IIII", data[offset : offset + 16]
            )
            offset += 16
            pkt_data = data[offset : offset + incl_len]
            offset += incl_len
            pkt_idx += 1

            if first_sec is None:
                first_sec = ts_sec
            rel_time = round((ts_sec - first_sec) + (ts_usec / 1e6), 4)

            # Ethernet header (14 bytes)
            if len(pkt_data) < 14:
                continue

            eth_type = struct.unpack("!H", pkt_data[12:14])[0]
            if eth_type != 0x0800:  # IPv4 only
                continue

            ip_header = pkt_data[14:]
            if len(ip_header) < 20:
                continue

            ihl = (ip_header[0] & 0x0F) * 4
            ttl = ip_header[8]
            protocol_num = ip_header[9]
            src_ip = ".".join(map(str, ip_header[12:16]))
            dst_ip = ".".join(map(str, ip_header[16:20]))

            proto = "TCP" if protocol_num == 6 else "UDP" if protocol_num == 17 else "ICMP" if protocol_num == 1 else "OTHER"

            sport = 0
            dport = 0
            flags_str = "NONE"

            transport = ip_header[ihl:]
            if protocol_num == 6 and len(transport) >= 14:  # TCP
                sport, dport = struct.unpack("!HH", transport[:4])
                flags_byte = transport[13]
                flags_list = []
                if flags_byte & 0x02: flags_list.append("SYN")
                if flags_byte & 0x10: flags_list.append("ACK")
                if flags_byte & 0x01: flags_list.append("FIN")
                if flags_byte & 0x04: flags_list.append("RST")
                if flags_byte & 0x08: flags_list.append("PSH")
                flags_str = "-".join(flags_list) if flags_list else "NONE"
            elif protocol_num == 17 and len(transport) >= 4:  # UDP
                sport, dport = struct.unpack("!HH", transport[:4])

            entropy = calculate_entropy(transport[20:] if len(transport) > 20 else b"")

            records.append({
                "packet_index": pkt_idx,
                "relative_time_sec": rel_time,
                "source_ip": src_ip,
                "destination_ip": dst_ip,
                "source_port": sport,
                "destination_port": dport,
                "protocol": proto,
                "packet_length": incl_len,
                "payload_size": max(0, incl_len - 14 - ihl),
                "time_to_live": ttl,
                "tcp_flags": flags_str,
                "shannon_entropy": entropy,
                "mac_address": "00:1B:44:11:3A:B7",
            })

        return records
