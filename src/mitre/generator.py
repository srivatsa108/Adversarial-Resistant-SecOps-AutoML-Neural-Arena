"""Automated MITRE ATT&CK Mapping & Live Defensive Rule Generator.

Integrates with Threat Intelligence Agent / Gemini API with an offline heuristic
cybersecurity rule synthesis fallback engine.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from src.config import GLOBAL_CONFIG
from src.interfaces.mitre import BaseMitreRuleGenerator
from src.security.privacy import PrivacyMasker
from src.utils.providers import PROVIDERS, key_from_environment


class MitreRuleGenerator(BaseMitreRuleGenerator):
    """Generates MITRE ATT&CK mappings, YARA rules, Snort/Suricata signatures, and firewall commands."""

    def __init__(self, privacy_masker: Optional[PrivacyMasker] = None) -> None:
        self._masker = privacy_masker or PrivacyMasker()

    def generate(
        self,
        model_summary: Dict[str, Any],
        top_features: List[Dict[str, Any]],
        anomalies_sample: Optional[List[Dict[str, Any]]] = None,
        robustness_metrics: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        provider: str = "Google Gemini",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate MITRE ATT&CK mappings and production defensive rules."""
        # 1. Package sanitized JSON payload
        raw_payload = {
            "model_metadata": model_summary,
            "top_attributions": top_features,
            "sample_anomalies": (anomalies_sample[:5] if anomalies_sample else []),
            "adversarial_robustness": robustness_metrics or {},
        }

        # Apply Zero-Trust masking to payload
        sanitized_payload_str = self._masker.mask_text(json.dumps(raw_payload, default=str))
        sanitized_payload = json.loads(sanitized_payload_str)

        # 2. Use the chosen provider when a key is available
        active_key = api_key or key_from_environment(provider)

        if active_key and active_key.strip():
            try:
                ai_output = self._generate_with_llm(sanitized_payload, active_key.strip(), provider, model, base_url)
                if ai_output:
                    ai_output["generator_mode"] = f"Threat intelligence agent ({provider})"
                    return ai_output
            except Exception:
                # Fall back to the local rule engine if the API fails or quota is exceeded
                pass

        # 3. Deterministic Local Heuristic Cyber Engine Fallback
        local_output = self._generate_local_defenses(sanitized_payload)
        local_output["generator_mode"] = "Local Cyber Heuristic Synthesis Engine"
        return local_output

    @staticmethod
    def _build_prompt(payload: Dict[str, Any]) -> str:
        return (
            "You are a Principal Cyber Defense Engineer and Threat Intelligence Specialist. "
            "Analyze this sanitized ML model decision boundary, SHAP feature explanations, "
            "and cybersecurity anomaly payload:\n\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Generate actionable defensive intelligence strictly in valid JSON format with the following keys:\n"
            "{\n"
            '  "mitre_mappings": [\n'
            '    {"tactic": "Initial Access | Discovery | Command and Control | etc", "technique_id": "T1xxx", "technique_name": "...", "justification": "..."}\n'
            "  ],\n"
            '  "yara_rule": "rule Cyber_Detection_Generated { ... }",\n'
            '  "snort_signature": "alert tcp any any -> any any (msg:...; sid:1000001; rev:1;)",\n'
            '  "firewall_commands": ["sudo iptables -A INPUT ...", "sudo ufw deny ..."],\n'
            '  "secops_action_plan": ["Step 1: ...", "Step 2: ...", "Step 3: ...", "Step 4: ..."]\n'
            "}\n"
            "Return ONLY the JSON object without markdown fences or additional prose."
        )

    @staticmethod
    def _parse_json_reply(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n", "", text)
            text = re.sub(r"\n```$", "", text).strip()
        return json.loads(text)

    def _generate_with_llm(
        self,
        payload: Dict[str, Any],
        api_key: str,
        provider: str,
        model: Optional[str],
        base_url: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Send the sanitized payload to the selected provider and return parsed JSON."""
        cfg = PROVIDERS.get(provider, PROVIDERS["Google Gemini"])
        kind = cfg["kind"]
        model = (model or str(cfg["model"])).strip()
        url = (base_url or str(cfg["base_url"])).strip().rstrip("/")
        prompt = self._build_prompt(payload)
        try:
            if kind == "gemini":
                from google import genai

                client = genai.Client(api_key=api_key)
                reply = client.models.generate_content(model=model or GLOBAL_CONFIG.DEFAULT_GEMINI_MODEL, contents=prompt)
                return self._parse_json_reply(reply.text)

            import requests

            if kind == "anthropic":
                r = requests.post(
                    f"{url}/v1/messages",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                    json={"model": model, "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]},
                    timeout=60,
                )
                r.raise_for_status()
                return self._parse_json_reply(r.json()["content"][0]["text"])

            if not url or not model:
                return None
            r = requests.post(
                f"{url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=60,
            )
            r.raise_for_status()
            return self._parse_json_reply(r.json()["choices"][0]["message"]["content"])
        except Exception:
            return None

    def _generate_with_gemini(self, payload: Dict[str, Any], api_key: str) -> Optional[Dict[str, Any]]:
        """Kept for backward compatibility with older callers."""
        return self._generate_with_llm(payload, api_key, "Google Gemini", None, None)

    def _generate_local_defenses(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize MITRE mappings, YARA rules, Snort signatures, and iptables locally."""
        top_feats = payload.get("top_attributions", [])
        anomalies = payload.get("sample_anomalies", [])
        feat_names = [f.get("feature", "").lower() for f in top_feats]
        all_text = " ".join(feat_names) + " " + json.dumps(anomalies).lower()

        # Heuristic mapping
        mitre_list: List[Dict[str, str]] = []
        if any(k in all_text for k in ("port", "scan", "syn", "sweep", "service")):
            mitre_list.append({
                "tactic": "Discovery (TA0007)",
                "technique_id": "T1046",
                "technique_name": "Network Service Discovery",
                "justification": "Telemetry reveals anomalous port enumeration and service interrogation patterns.",
            })
        if any(k in all_text for k in ("url", "phish", "email", "link", "domain")):
            mitre_list.append({
                "tactic": "Initial Access (TA0001)",
                "technique_id": "T1566",
                "technique_name": "Phishing",
                "justification": "High SHAP attribution on suspicious URL characteristics and email delivery tokens.",
            })
        if any(k in all_text for k in ("bytes", "volume", "flood", "ddos", "rate")):
            mitre_list.append({
                "tactic": "Impact (TA0040)",
                "technique_id": "T1499",
                "technique_name": "Endpoint Denial of Service",
                "justification": "Significant volumetric anomalies deviating substantially from standard packet quotas.",
            })
        if any(k in all_text for k in ("rdp", "ssh", "login", "auth", "credential", "failed")):
            mitre_list.append({
                "tactic": "Credential Access (TA0006)",
                "technique_id": "T1110",
                "technique_name": "Brute Force",
                "justification": "Repeated authentication deviations and unauthorized administrative access attempts.",
            })
        if not mitre_list:
            mitre_list.append({
                "tactic": "Command and Control (TA0011)",
                "technique_id": "T1071",
                "technique_name": "Application Layer Protocol",
                "justification": "Deep Autoencoder highlighted multi-attribute protocol structural deviations.",
            })

        # Generate YARA Rule
        primary_feat = top_feats[0].get("feature", "traffic_anomaly") if top_feats else "threat_signature"
        yara_rule = (
            "rule SecOps_AutoML_Detection {\n"
            "    meta:\n"
            '        description = "Automated defense signature generated by SecOps AutoML Engine"\n'
            '        author = "Adversa SecOps Engine"\n'
            '        severity = "High"\n'
            '        date = "2026-11-06"\n'
            "    strings:\n"
            f'        $sig_feature1 = "{primary_feat}" ascii wide nocase\n'
            '        $syn_flood = { 02 04 ?? ?? 01 03 03 }\n'
            '        $anomaly_token = "INTERNAL_HOST" ascii\n'
            "    condition:\n"
            "        2 of them and filesize < 10MB\n"
            "}"
        )

        # Generate Snort / Suricata IDS Signature
        target_port = "any"
        for word in all_text.split():
            if word.isdigit() and int(word) in (22, 80, 443, 3389, 8080):
                target_port = word
                break
        if target_port == "any" and "3389" in all_text:
            target_port = "3389"

        snort_sig = (
            f'alert tcp $EXTERNAL_NET any -> $HOME_NET {target_port} '
            f'(msg:"SECOPS-AUTOML: Statistical Anomaly Surpassed Baseline Threshold"; '
            f'flow:to_server,established; content:"|00|"; depth:4; '
            f'classtype:suspicious-traffic; sid:2026001; rev:1;)'
        )

        # Generate Linux iptables & ufw block commands
        firewall_cmds = [
            "# 1. Block high-risk external anomaly subnets",
            "sudo iptables -A INPUT -p tcp --dport 3389 -m state --state NEW -m recent --set",
            "sudo iptables -A INPUT -p tcp --dport 3389 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP",
            "# 2. Enforce strict UFW rate-limiting on ingress telemetry ports",
            "sudo ufw limit proto tcp to any port 22 comment 'Rate limit SSH brute force'",
            "sudo ufw deny from 198.51.100.0/24 to any comment 'Block adversary evasion subnet'",
        ]

        # SecOps Action Plan
        action_plan = [
            "1. Isolate the affected endpoint / network segment tagged by the Deep Autoencoder.",
            "2. Deploy the synthesized Snort/Suricata signature (SID 2026001) to boundary NIDS appliances.",
            "3. Enforce the iptables connection rate-limiting rule to mitigate brute-force and evasion mutations.",
            "4. Perform adversarial data augmentation and retrain the winning model to harden detection boundaries.",
        ]

        return {
            "mitre_mappings": mitre_list,
            "yara_rule": yara_rule,
            "snort_signature": snort_sig,
            "firewall_commands": firewall_cmds,
            "secops_action_plan": action_plan,
        }
