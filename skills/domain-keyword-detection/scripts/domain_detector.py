#!/usr/bin/env python3
"""
Domain Keyword Detection Skill

Detect document domain (software/hardware/manufacturing/standards/business/biotech)
via lightweight keyword matching.

Usage:
    python3 domain_detector.py detect --input <file> [--threshold 0.3] [--output json]
    python3 domain_detector.py get-keywords --domain <domain_name>
    python3 domain_detector.py list-domains
    python3 domain_detector.py validate
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any
import re

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Install with: pip3 install pyyaml", file=sys.stderr)
    sys.exit(2)


class DomainDetector:
    """Detect document domain via keyword matching."""
    
    def __init__(self, domains_dir: Path = None):
        """Initialize detector with domain configuration directory."""
        if domains_dir is None:
            # Default: domains/ subdirectory relative to this script
            script_dir = Path(__file__).parent
            domains_dir = script_dir.parent / "domains"
        
        self.domains_dir = Path(domains_dir)
        self.domains: Dict[str, Dict] = {}
        self._load_domains()
    
    def _load_domains(self):
        """Load all domain YAML files from domains/ directory."""
        if not self.domains_dir.exists():
            raise FileNotFoundError(f"Domains directory not found: {self.domains_dir}")
        
        for yaml_file in self.domains_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    domain_config = yaml.safe_load(f)
                    domain_name = domain_config.get('name')
                    if not domain_name:
                        print(f"Warning: {yaml_file.name} missing 'name' field, skipping", file=sys.stderr)
                        continue
                    self.domains[domain_name] = domain_config
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file.name}: {e}", file=sys.stderr)
    
    def _extract_all_keywords(self, domain_config: Dict) -> Set[str]:
        """Extract all keywords from a domain config (flattened)."""
        keywords = set()
        keywords_dict = domain_config.get('keywords', {})
        
        def flatten(d):
            """Recursively flatten nested dict/list to extract string keywords."""
            if isinstance(d, dict):
                for v in d.values():
                    flatten(v)
            elif isinstance(d, list):
                for item in d:
                    if isinstance(item, str):
                        keywords.add(item.lower())
                    else:
                        flatten(item)
            elif isinstance(d, str):
                keywords.add(d.lower())
        
        flatten(keywords_dict)
        return keywords
    
    def _match_keywords_in_text(self, text_lower: str, keywords: Set[str]) -> Set[str]:
        """
        Match keywords in text using substring matching (supports CJK).
        This allows partial matching: "纳米晶材料" matches keyword "纳米晶".
        """
        matched = set()
        for keyword in keywords:
            # Substring match for CJK and compound terms
            if keyword in text_lower:
                matched.add(keyword)
        return matched
    
    def detect(self, text: str, threshold: float = 0.3) -> Dict[str, Any]:
        """
        Detect domain(s) from text.
        
        Args:
            text: Document text (Markdown, plain text, etc.)
            threshold: Minimum confidence score (0.0-1.0) to activate a domain
        
        Returns:
            {
                "status": "success" | "empty" | "error",
                "detected_domains": [str],
                "confidence_scores": {domain: float},
                "matched_keywords": {domain: [str]},
                "activated_packs": [str],
                "threshold": float,
                "total_keywords_matched": int
            }
        """
        if not text or not text.strip():
            return {
                "status": "empty",
                "detected_domains": [],
                "confidence_scores": {},
                "threshold": threshold,
                "hint": "Empty input text"
            }
        
        text_lower = text.lower()
        
        # Calculate document length (CJK-aware: count CJK chars + English words)
        cjk_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_words = len(re.findall(r'[a-zA-Z]+', text))
        doc_length_units = cjk_chars + en_words
        
        results = {}
        
        for domain_name, domain_config in self.domains.items():
            domain_keywords = self._extract_all_keywords(domain_config)
            matched = self._match_keywords_in_text(text_lower, domain_keywords)
            
            if not matched:
                continue
            
            # Calculate confidence score
            base_score = len(matched) / len(domain_keywords)
            
            # Apply weight factors
            weight = 1.0
            
            # Penalty for short documents (reduce false positives)
            # Use CJK-aware length units
            if doc_length_units < 50:
                weight *= 0.5
            elif doc_length_units < 200:
                weight *= 0.7
            
            # Boost for strong domain signals
            if len(matched) > 15:
                weight *= 1.2
            elif len(matched) > 8:
                weight *= 1.1
            
            # Apply category weights from config
            weights = domain_config.get('scoring_weights', {})
            avg_weight = sum(weights.values()) / len(weights) if weights else 1.0
            weight *= avg_weight
            
            confidence = min(base_score * weight, 1.0)
            
            if confidence >= threshold:
                results[domain_name] = {
                    "confidence": confidence,
                    "matched_keywords": sorted(matched)
                }
        
        if not results:
            return {
                "status": "empty",
                "detected_domains": [],
                "confidence_scores": {},
                "threshold": threshold,
                "hint": f"No domains matched above threshold {threshold}. Try lowering threshold or check document content."
            }
        
        # Sort by confidence
        sorted_domains = sorted(results.items(), key=lambda x: x[1]['confidence'], reverse=True)
        
        return {
            "status": "success",
            "detected_domains": [d[0] for d in sorted_domains],
            "confidence_scores": {d[0]: round(d[1]['confidence'], 3) for d in sorted_domains},
            "matched_keywords": {d[0]: d[1]['matched_keywords'] for d in sorted_domains},
            "activated_packs": [d[0] for d in sorted_domains],
            "threshold": threshold,
            "total_keywords_matched": sum(len(d[1]['matched_keywords']) for d in sorted_domains)
        }
    
    def get_keywords(self, domain: str) -> Dict[str, Any]:
        """Get keyword list for a specific domain."""
        if domain not in self.domains:
            return {
                "status": "error",
                "error_message": f"Domain '{domain}' not found",
                "hint": f"Available domains: {', '.join(self.domains.keys())}"
            }
        
        domain_config = self.domains[domain]
        keywords_dict = domain_config.get('keywords', {})
        
        return {
            "status": "success",
            "domain": domain,
            "keywords": keywords_dict
        }
    
    def list_domains(self) -> Dict[str, Any]:
        """List all available domains."""
        return {
            "status": "success",
            "available_domains": sorted(self.domains.keys()),
            "domain_descriptions": {
                name: config.get('description', 'No description')
                for name, config in self.domains.items()
            }
        }
    
    def validate(self) -> Dict[str, Any]:
        """Validate all domain YAML files."""
        valid_domains = []
        invalid_domains = {}
        
        for domain_name, domain_config in self.domains.items():
            errors = []
            
            # Check required fields
            required_fields = ['name', 'description', 'keywords']
            for field in required_fields:
                if field not in domain_config:
                    errors.append(f"Missing required field: {field}")
            
            # Check keywords structure
            if 'keywords' in domain_config:
                keywords_dict = domain_config['keywords']
                if not isinstance(keywords_dict, dict):
                    errors.append("'keywords' must be a dictionary")
                elif len(keywords_dict) == 0:
                    errors.append("'keywords' is empty")
            
            if errors:
                invalid_domains[domain_name] = {"errors": errors}
            else:
                valid_domains.append(domain_name)
        
        status = "success" if not invalid_domains else "error"
        
        return {
            "status": status,
            "valid_domains": sorted(valid_domains),
            "invalid_domains": invalid_domains
        }


def main():
    parser = argparse.ArgumentParser(description="Domain Keyword Detection")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # detect command
    detect_parser = subparsers.add_parser('detect', help='Detect domain(s) from document')
    detect_parser.add_argument('--input', '-i', required=True, help='Input file path or "-" for stdin')
    detect_parser.add_argument('--threshold', '-t', type=float, default=0.3, help='Confidence threshold (0.0-1.0)')
    detect_parser.add_argument('--output', '-o', choices=['json', 'yaml'], default='json', help='Output format')
    
    # get-keywords command
    keywords_parser = subparsers.add_parser('get-keywords', help='Get keywords for a domain')
    keywords_parser.add_argument('--domain', '-d', required=True, help='Domain name')
    
    # list-domains command
    subparsers.add_parser('list-domains', help='List all available domains')
    
    # validate command
    subparsers.add_parser('validate', help='Validate domain configuration files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        detector = DomainDetector()
        
        if args.command == 'detect':
            # Read input
            if args.input == '-':
                text = sys.stdin.read()
            else:
                input_path = Path(args.input)
                if not input_path.exists():
                    print(json.dumps({
                        "status": "error",
                        "error_message": f"Input file not found: {args.input}"
                    }))
                    sys.exit(1)
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            result = detector.detect(text, threshold=args.threshold)
            
        elif args.command == 'get-keywords':
            result = detector.get_keywords(args.domain)
            
        elif args.command == 'list-domains':
            result = detector.list_domains()
            
        elif args.command == 'validate':
            result = detector.validate()
        
        # Output result
        if args.command == 'detect' and args.output == 'yaml':
            print(yaml.dump(result, allow_unicode=True, default_flow_style=False))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # Exit code
        if result.get('status') == 'error':
            sys.exit(1)
        else:
            sys.exit(0)
    
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error_message": str(e),
            "hint": "Check input file and domain configuration"
        }), file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
