"""Abstract Base Class for MITRE ATT&CK mapping and automated defensive rule generation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMitreRuleGenerator(ABC):
    """Abstract interface defining MITRE mapping and defensive rule synthesis."""

    @abstractmethod
    def generate(
        self,
        model_summary: Dict[str, Any],
        top_features: List[Dict[str, Any]],
        anomalies_sample: Optional[List[Dict[str, Any]]] = None,
        robustness_metrics: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate MITRE ATT&CK mapping, YARA rule, Snort signature, and firewall block commands.

        Args:
            model_summary: High-level model metadata and performance metrics.
            top_features: Identified high-impact cyber features and thresholds.
            anomalies_sample: Extracted high-risk anomaly event vectors.
            robustness_metrics: Adversarial vulnerability metrics.
            api_key: Optional Gemini API Key for Threat Intelligence Agent.

        Returns:
            Dictionary containing:
                - 'mitre_mappings': List of Tactic & Technique objects with IDs,
                - 'yara_rule': Synthesized YARA detection rule string,
                - 'snort_signature': Synthesized Snort/Suricata IDS rule string,
                - 'firewall_commands': Linux iptables / ufw block commands,
                - 'secops_action_plan': Step-by-step SOC analyst remediation checklist.
        """
        pass
