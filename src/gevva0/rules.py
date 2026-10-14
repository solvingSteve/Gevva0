"""Tier -1 Rule-Based Heuristic Routing System.

Provides ultra-low-latency (< 50 microseconds) deterministic rule evaluation
before neural routing layers (Tier 0 and Tier 1). Rules are defined in JSON
configuration files in the `RoutingRules/` folder and can be hot-reloaded
at runtime without model re-initialization.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from .config import PROJECT_ROOT

logger = logging.getLogger("gevva0.rules")

ROUTING_RULES_DIR = PROJECT_ROOT / "RoutingRules"


class RulePredicate:
    """Pre-compiled rule predicate evaluating in sub-50 microseconds."""

    def __init__(self, rule_dict: dict[str, Any]):
        self.name: str = rule_dict.get("name", "Unnamed Rule")
        self.description: str = rule_dict.get("description", "")
        self.action: str = rule_dict.get("action", "EVALUATE_ALL").strip().upper()
        self.priority: int = rule_dict.get("priority", 99)
        self.decision: str | None = rule_dict.get("decision")  # optional direct target label

        conds = rule_dict.get("conditions", {})
        self.min_words: int = int(conds.get("min_words", 0))
        max_w = conds.get("max_words")
        self.max_words: float = float(max_w) if max_w is not None else float("inf")

        # Pre-compile regex once on load
        raw_regex = conds.get("regex_match")
        self.raw_regex: str | None = raw_regex
        self.regex: re.Pattern | None = (
            re.compile(raw_regex, re.IGNORECASE) if raw_regex else None
        )

        # Lowercase keywords for instant substring checks
        self.keywords_any: list[str] = [
            str(k).lower().strip() for k in conds.get("keywords_any", []) if str(k).strip()
        ]
        self.keywords_all: list[str] = [
            str(k).lower().strip() for k in conds.get("keywords_all", []) if str(k).strip()
        ]
        self.target_option_keywords: list[str] = [
            str(k).lower().strip() for k in conds.get("target_option_keywords", []) if str(k).strip()
        ]
        raw_opt_regex = conds.get("option_regex")
        self.raw_option_regex: str | None = raw_opt_regex
        self.option_regex: re.Pattern | None = (
            re.compile(raw_opt_regex, re.IGNORECASE) if raw_opt_regex else None
        )

    def matches(self, text: str, word_count: int) -> bool:
        """Evaluates whether the given text matches this rule's conditions."""
        if word_count < self.min_words or word_count > self.max_words:
            return False

        text_lower = text.lower()

        # Check keywords_all if specified
        if self.keywords_all:
            if not all(k in text_lower for k in self.keywords_all):
                return False

        # If keywords_any specified, at least one must match
        if self.keywords_any:
            if any(k in text_lower for k in self.keywords_any):
                return True

        # If regex specified, pattern must match
        if self.regex and self.regex.search(text):
            return True

        # If conditions only specified word count or keywords_all (no regex or keywords_any), match
        if not self.keywords_any and not self.regex:
            return True

        return False

    def to_dict(self) -> dict[str, Any]:
        conds: dict[str, Any] = {}
        if self.min_words > 0:
            conds["min_words"] = self.min_words
        if self.max_words != float("inf"):
            conds["max_words"] = int(self.max_words)
        if self.raw_regex:
            conds["regex_match"] = self.raw_regex
        if self.keywords_any:
            conds["keywords_any"] = self.keywords_any
        if self.keywords_all:
            conds["keywords_all"] = self.keywords_all
        if self.target_option_keywords:
            conds["target_option_keywords"] = self.target_option_keywords
        if self.raw_option_regex:
            conds["option_regex"] = self.raw_option_regex

        res = {
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "priority": self.priority,
            "conditions": conds,
        }
        if self.decision:
            res["decision"] = self.decision
        return res


class RuleMatchResult:
    """Outcome of evaluating Tier -1 rules against a context."""

    def __init__(
        self,
        matched: bool,
        rule_name: str | None = None,
        action: str = "EVALUATE_ALL",
        description: str = "",
        priority: int = 99,
        decision: str | None = None,
        decision_text: str | None = None,
    ):
        self.matched = matched
        self.rule_name = rule_name
        self.action = action
        self.description = description
        self.priority = priority
        self.decision = decision
        self.decision_text = decision_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "rule_name": self.rule_name,
            "action": self.action,
            "description": self.description,
            "priority": self.priority,
            "decision": self.decision,
            "decision_text": self.decision_text,
        }


class ConfigurableGateway:
    """Tier -1 Rule-based gateway managing JSON rule sets and sub-millisecond evaluation."""

    def __init__(
        self,
        rules_dir: Path | str | None = None,
        active_file: str = "routing_rules.json",
    ):
        self.rules_dir = Path(rules_dir) if rules_dir else ROUTING_RULES_DIR
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        self.active_file: str = active_file
        self.version: str = "1.0"
        self.default_action: str = "EVALUATE_ALL"
        self.rules: list[RulePredicate] = []
        self.load_rules(self.active_file)

    @property
    def config_path(self) -> Path:
        return self.rules_dir / self.active_file

    def list_rule_files(self) -> list[dict[str, Any]]:
        """Scans the RoutingRules folder and returns metadata for all .json files."""
        if not self.rules_dir.is_dir():
            return []

        results: list[dict[str, Any]] = []
        for p in sorted(self.rules_dir.glob("*.json")):
            meta = {
                "filename": p.name,
                "is_active": p.name == self.active_file,
                "rule_count": 0,
                "version": "1.0",
                "default_action": "EVALUATE_ALL",
                "rules": [],
            }
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                meta["version"] = data.get("version", "1.0")
                meta["default_action"] = data.get("default_action", "EVALUATE_ALL")
                raw_rules = data.get("rules", [])
                meta["rule_count"] = len(raw_rules)
                meta["rules"] = raw_rules
            except Exception as e:
                meta["error"] = str(e)
            results.append(meta)
        return results

    def load_rules(self, filename: str | None = None) -> int:
        """Loads and pre-compiles rules from the specified JSON file or active file."""
        if filename:
            self.active_file = filename

        target_p = self.config_path
        if not target_p.exists():
            print(f"[Tier -1] Config {target_p} not found. Running with no heuristic rules.")
            self.rules = []
            return 0

        try:
            with open(target_p, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.version = data.get("version", "1.0")
            self.default_action = data.get("default_action", "EVALUATE_ALL").strip().upper()
            raw_rules = sorted(data.get("rules", []), key=lambda r: r.get("priority", 99))
            self.rules = [RulePredicate(r) for r in raw_rules]
            print(f"[Tier -1] Successfully loaded and compiled {len(self.rules)} rules from {target_p.name}")
            return len(self.rules)
        except Exception as exc:
            print(f"[Tier -1] Error parsing {target_p}: {exc}")
            self.rules = []
            return 0

    def evaluate_route(
        self,
        context: str,
        options: dict[str, str] | list[str] | None = None,
    ) -> RuleMatchResult | None:
        """Evaluates rules sequentially in under 0.05 ms.

        Returns a RuleMatchResult if a rule matched, or None if no rule matched.
        """
        clean_text = context.strip()
        if not clean_text:
            return None

        word_count = len(clean_text.split())

        for rule in self.rules:
            if rule.matches(clean_text, word_count):
                resolved_decision: str | None = rule.decision
                resolved_text: str | None = None
                requires_option_match = bool(rule.target_option_keywords or rule.option_regex)

                if options:
                    if isinstance(options, dict):
                        if requires_option_match:
                            matched_opt = False
                            for letter, opt_txt in options.items():
                                opt_low = opt_txt.lower().strip()
                                let_low = letter.lower().strip()
                                if rule.option_regex and (
                                    rule.option_regex.search(opt_txt) or rule.option_regex.search(letter)
                                ):
                                    resolved_decision = letter
                                    resolved_text = opt_txt
                                    matched_opt = True
                                    break
                                if rule.target_option_keywords and any(
                                    k == opt_low or k == let_low or k in opt_low
                                    for k in rule.target_option_keywords
                                ):
                                    resolved_decision = letter
                                    resolved_text = opt_txt
                                    matched_opt = True
                                    break
                            if not matched_opt:
                                continue
                        elif resolved_decision and resolved_decision in options:
                            resolved_text = options[resolved_decision]
                        elif not resolved_decision:
                            # Try to match keywords to option labels
                            for letter, opt_txt in options.items():
                                if any(k in opt_txt.lower() for k in rule.keywords_any):
                                    resolved_decision = letter
                                    resolved_text = opt_txt
                                    break
                            if not resolved_decision:
                                resolved_decision = next(iter(options.keys()))
                                resolved_text = options[resolved_decision]
                    elif isinstance(options, list):
                        if requires_option_match:
                            matched_opt = False
                            for opt_txt in options:
                                opt_low = str(opt_txt).lower().strip()
                                if rule.option_regex and rule.option_regex.search(str(opt_txt)):
                                    resolved_decision = str(opt_txt)
                                    resolved_text = str(opt_txt)
                                    matched_opt = True
                                    break
                                if rule.target_option_keywords and any(
                                    k == opt_low or k in opt_low for k in rule.target_option_keywords
                                ):
                                    resolved_decision = str(opt_txt)
                                    resolved_text = str(opt_txt)
                                    matched_opt = True
                                    break
                            if not matched_opt:
                                continue
                        elif resolved_decision and resolved_decision in options:
                            resolved_text = resolved_decision
                        elif not resolved_decision and options:
                            resolved_decision = options[0]
                            resolved_text = options[0]
                elif requires_option_match and rule.action == "DIRECT_EXIT":
                    # Cannot resolve option-dependent DIRECT_EXIT without options
                    continue

                return RuleMatchResult(
                    matched=True,
                    rule_name=rule.name,
                    action=rule.action,
                    description=rule.description,
                    priority=rule.priority,
                    decision=resolved_decision,
                    decision_text=resolved_text,
                )

        return None


_GLOBAL_GATEWAY: ConfigurableGateway | None = None


def get_rule_gateway(rules_dir: Path | str | None = None) -> ConfigurableGateway:
    """Returns global singleton ConfigurableGateway."""
    global _GLOBAL_GATEWAY
    if _GLOBAL_GATEWAY is None:
        _GLOBAL_GATEWAY = ConfigurableGateway(rules_dir=rules_dir)
    return _GLOBAL_GATEWAY
