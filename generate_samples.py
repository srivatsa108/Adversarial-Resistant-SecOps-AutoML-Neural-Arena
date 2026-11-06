"""Generate realistic cybersecurity sample datasets for demonstration and testing.

Outputs:
1. data/samples/phishing_data.csv: Messy supervised classification dataset with duplicates,
   missing values, junk metadata, and emails.
2. data/samples/firewall_logs.json: Unsupervised network log array containing internal RFC1918
   IPs (e.g., 192.168.1.50), zero-day port scans, and RDP volumetric anomalies.
"""

import json
import os
import random
import numpy as np
import pandas as pd


def generate_samples(output_dir: str = "data/samples") -> None:
    """Generate sample cybersecurity datasets in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    # ==========================================
    # 1. Supervised Dataset: phishing_data.csv
    # ==========================================
    n_phishing = 300
    rows = []

    for i in range(n_phishing):
        is_phish = random.random() < 0.35
        label = "Phishing" if is_phish else "Legitimate"

        url_length = int(np.random.normal(85, 25)) if is_phish else int(np.random.normal(35, 10))
        url_length = max(15, url_length)

        has_ip = 1 if (is_phish and random.random() < 0.45) else 0
        special_chars = int(np.random.poisson(6 if is_phish else 1))
        subdomains = int(np.random.poisson(3 if is_phish else 1))
        suspicious_words = int(np.random.poisson(2 if is_phish else 0.2))
        ssl_age_days = int(np.random.exponential(30 if is_phish else 600))
        entropy = round(float(np.random.normal(4.8 if is_phish else 3.2, 0.4)), 3)

        email_user = f"user_{random.randint(10, 99)}"
        email_domain = "corp.internal" if random.random() < 0.5 else "external-sec.org"
        sender_email = f"{email_user}@{email_domain}"

        # Deliberate junk column with 0 variance
        junk_constant = "GLOBAL_SCANNER_v2.1"

        # Missing value injection
        if random.random() < 0.08:
            ssl_age_days = None

        rows.append({
            "url_length": url_length,
            "has_ip_in_url": has_ip,
            "special_char_count": special_chars,
            "subdomain_count": subdomains,
            "suspicious_keywords_count": suspicious_words,
            "ssl_age_days": ssl_age_days,
            "shannon_entropy": entropy,
            "sender_email": sender_email,
            "junk_scanner_tag": junk_constant,
            "threat_label": label,
        })

    # Add duplicate rows deliberately
    rows.append(rows[0].copy())
    rows.append(rows[5].copy())
    rows.append(rows[10].copy())

    phishing_df = pd.DataFrame(rows)
    phishing_path = os.path.join(output_dir, "phishing_data.csv")
    phishing_df.to_csv(phishing_path, index=False)
    print(f"Generated {len(phishing_df)} phishing records -> {phishing_path}")

    # ==========================================
    # 2. Unsupervised Dataset: firewall_logs.json
    # ==========================================
    n_logs = 400
    logs = []

    internal_ips = ["192.168.1.50", "192.168.1.102", "10.0.0.15", "10.0.4.22", "172.16.8.5"]
    external_ips = ["203.0.113.19", "198.51.100.44", "198.51.100.89", "8.8.8.8", "1.1.1.1"]

    for i in range(n_logs):
        # 95% normal traffic, 5% zero-day anomalies
        is_anomaly = i >= (n_logs - 20)

        if not is_anomaly:
            src = random.choice(internal_ips)
            dst = random.choice(external_ips)
            dport = random.choice([80, 443, 8080, 53])
            proto = "TCP" if dport != 53 else "UDP"
            pkt_len = int(np.random.normal(512, 120))
            duration = round(float(np.random.exponential(1.5)), 3)
            bytes_sent = int(pkt_len * np.random.randint(5, 30))
            flags = random.choice(["ACK", "SYN-ACK", "PSH-ACK"])
        else:
            # Zero-Day Anomaly: Stealthy Port Scan / RDP volumetric flood
            src = "192.168.1.50"  # Specific internal host under compromise
            dst = "198.51.100.44"
            dport = 3389 if random.random() < 0.6 else random.randint(1024, 65535)
            proto = "TCP"
            pkt_len = int(np.random.normal(2400, 300))  # Major deviation
            duration = round(float(np.random.normal(45.0, 10.0)), 3)  # Long persistence
            bytes_sent = int(pkt_len * np.random.randint(150, 600))
            flags = "SYN"  # Port sweep / flood pattern

        logs.append({
            "source_ip": src,
            "destination_ip": dst,
            "destination_port": dport,
            "protocol": proto,
            "packet_length": max(40, pkt_len),
            "connection_duration_sec": max(0.01, duration),
            "bytes_transferred": max(100, bytes_sent),
            "tcp_flags": flags,
            "mac_address": "00:1B:44:11:3A:B7" if src == "192.168.1.50" else "70:85:C2:55:A1:00",
            "request_uri": f"https://{dst}/service/gateway" if dport in (80, 443) else f"tcp://{dst}:{dport}",
        })

    firewall_path = os.path.join(output_dir, "firewall_logs.json")
    with open(firewall_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    print(f"Generated {len(logs)} firewall records -> {firewall_path}")

    # ==========================================
    # 3. Network PCAP Dataset: network_traffic.pcap
    # ==========================================
    pcap_path = os.path.join(output_dir, "network_traffic.pcap")
    try:
        from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap

        packets = []
        base_time = 1700000000.0

        for idx in range(250):
            is_anomaly = idx >= 230  # Last 20 packets are port scan / RDP flood anomalies
            pkt_time = base_time + idx * 0.05

            if not is_anomaly:
                src_ip = random.choice(internal_ips)
                dst_ip = random.choice(external_ips)
                dport = random.choice([80, 443, 53])
                sport = random.randint(1024, 65535)

                if dport == 53:
                    pkt = Ether(src="00:1B:44:11:3A:B7", dst="70:85:C2:55:A1:00") / \
                          IP(src=src_ip, dst=dst_ip, ttl=64) / \
                          UDP(sport=sport, dport=53) / \
                          Raw(load=b"\x00\x01\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07gateway\x04corp\x00\x00\x01\x00\x01")
                else:
                    pkt = Ether(src="00:1B:44:11:3A:B7", dst="70:85:C2:55:A1:00") / \
                          IP(src=src_ip, dst=dst_ip, ttl=64) / \
                          TCP(sport=sport, dport=dport, flags="PA") / \
                          Raw(load=b"GET /index.html HTTP/1.1\r\nHost: corporate.internal\r\n\r\n")
            else:
                # Malicious SYN scan / RDP flood
                src_ip = "192.168.1.50"
                dst_ip = "198.51.100.44"
                target_port = 3389 if idx % 2 == 0 else (1000 + idx * 10)
                pkt = Ether(src="00:1B:44:11:3A:B7", dst="70:85:C2:55:A1:00") / \
                      IP(src=src_ip, dst=dst_ip, ttl=48) / \
                      TCP(sport=55442, dport=target_port, flags="S") / \
                      Raw(load=b"\x00" * random.randint(500, 1500))

            pkt.time = pkt_time
            packets.append(pkt)

        wrpcap(pcap_path, packets)
        print(f"Generated {len(packets)} raw packets -> {pcap_path}")
    except Exception as e:
        print(f"Could not generate scapy PCAP: {e}")


if __name__ == "__main__":
    generate_samples()
