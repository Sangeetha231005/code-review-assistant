"""
SECURITY SCANNER MODULE
Layer 3: Security Linters (STRONGLY RECOMMENDED)
Purpose: Catch known vulnerability patterns per language
Language-specific security tools for Python, Java, JavaScript, PHP, Ruby, Go
"""

import subprocess
import tempfile
import os
import re
import json
import shutil
import sys
from typing import Dict, List, Any, Optional, Tuple


class SecurityScanner:
    """
    Security Scanner for 6 programming languages.
    Implements industry-standard security tools for each language.
    """

    def __init__(self):
        # Language-specific security tools configuration
        self.security_tools = {
            "python": self._scan_python,
            "java": self._scan_java,
            "javascript": self._scan_javascript,
            "php": self._scan_php,
            "ruby": self._scan_ruby,
            "go": self._scan_go
        }

        # Security severity levels
        self.severity_levels = {
            "CRITICAL": 5,
            "HIGH": 4,
            "MEDIUM": 3,
            "LOW": 2,
            "INFO": 1
        }

        # Common vulnerability patterns (regex-based checks)
        self.vulnerability_patterns = {
            "python": [
                # SQL Injection
                (r"execute\(.*\+.*\)", "HIGH", "Possible SQL injection (string concatenation in execute())"),
                (r"executemany\(.*\+.*\)", "HIGH", "Possible SQL injection (string concatenation in executemany())"),
                # Command injection
                (r"os\.system\(.*\)", "CRITICAL", "Possible command injection in os.system()"),
                (r"subprocess\.call\(.*shell=True.*\)", "CRITICAL", "Possible command injection with shell=True"),
                (r"subprocess\.Popen\(.*shell=True.*\)", "CRITICAL", "Possible command injection with shell=True"),
                # Deserialization
                (r"pickle\.loads\(", "CRITICAL", "Unsafe deserialization with pickle"),
                (r"yaml\.load\(", "HIGH", "Unsafe YAML deserialization (use yaml.safe_load)"),
                # Insecure randomness
                (r"random\.randint\(", "MEDIUM", "Insecure random number for cryptography"),
                (r"random\.choice\(", "MEDIUM", "Insecure random for security contexts"),
                # Hardcoded secrets
                (r"password\s*=\s*['\"].{4,}['\"]", "HIGH", "Possible hardcoded password"),
                (r"secret\s*=\s*['\"].{4,}['\"]", "HIGH", "Possible hardcoded secret"),
                (r"api_key\s*=\s*['\"].{8,}['\"]", "HIGH", "Possible hardcoded API key"),
                # Insecure hashing
                (r"hashlib\.md5\(", "MEDIUM", "Weak MD5 hashing algorithm"),
                (r"hashlib\.sha1\(", "MEDIUM", "Weak SHA-1 hashing algorithm"),
                # SSL/TLS issues
                (r"verify\s*=\s*False", "HIGH", "SSL certificate verification disabled"),
            ],

            "javascript": [
                # DOM XSS
                (r"innerHTML\s*=", "HIGH", "Unsafe innerHTML assignment (potential XSS)"),
                (r"outerHTML\s*=", "HIGH", "Unsafe outerHTML assignment (potential XSS)"),
                (r"document\.write\(", "HIGH", "Unsafe document.write() (potential XSS)"),
                # Eval usage
                (r"eval\(", "CRITICAL", "Unsafe eval() usage"),
                (r"Function\(['\"].*['\"]\)", "CRITICAL", "Unsafe Function constructor"),
                # Insecure randomness
                (r"Math\.random\(\)", "MEDIUM", "Insecure random for security contexts"),
                # LocalStorage with sensitive data
                (r"localStorage\.setItem\(.*password.*\)", "HIGH", "Sensitive data in localStorage"),
                (r"localStorage\.setItem\(.*token.*\)", "HIGH", "Sensitive data in localStorage"),
                # JWT issues
                (r"jwt\.decode\(.*\{.*verify:\s*false.*\}\)", "HIGH", "JWT verification disabled"),
                # Prototype pollution
                (r"__proto__", "HIGH", "Prototype pollution risk"),
                (r"constructor\.prototype", "HIGH", "Prototype pollution risk"),
            ],

            "java": [
                # SQL Injection
                (r"\.executeQuery\(.*\+.*\)", "HIGH", "Possible SQL injection (string concatenation)"),
                (r"\.executeUpdate\(.*\+.*\)", "HIGH", "Possible SQL injection (string concatenation)"),
                # Command injection
                (r"Runtime\.getRuntime\(\)\.exec\(", "CRITICAL", "Command execution without validation"),
                # Deserialization
                (r"ObjectInputStream", "CRITICAL", "Unsafe deserialization"),
                (r"readObject\(", "CRITICAL", "Unsafe deserialization"),
                # Insecure randomness
                (r"java\.util\.Random", "MEDIUM", "Insecure random for cryptography"),
                # Hardcoded secrets
                (r"String\s+password\s*=\s*\".*\"", "HIGH", "Possible hardcoded password"),
                (r"String\s+secret\s*=\s*\".*\"", "HIGH", "Possible hardcoded secret"),
                (r"String\s+apiKey\s*=\s*\".*\"", "HIGH", "Possible hardcoded API key"),
                # Cryptographic issues
                (r"MD5", "MEDIUM", "Weak MD5 algorithm"),
                (r"SHA1", "MEDIUM", "Weak SHA-1 algorithm"),
                # SSL/TLS issues
                (r"setVerifyMode\(.*NO_VERIFY", "HIGH", "SSL verification disabled"),
            ],

            "php": [
                # SQL Injection
                (r"mysql_query\(\$.*\.\$.*\)", "CRITICAL", "Possible SQL injection (string concatenation)"),
                (r"mysqli_query\(\$.*\.\$.*\)", "CRITICAL", "Possible SQL injection (string concatenation)"),
                # Command injection
                (r"exec\(\$.*\)", "CRITICAL", "Command injection risk"),
                (r"shell_exec\(\$.*\)", "CRITICAL", "Command injection risk"),
                (r"system\(\$.*\)", "CRITICAL", "Command injection risk"),
                # File inclusion
                (r"include\(\$.*\)", "HIGH", "Dynamic file inclusion risk"),
                (r"require\(\$.*\)", "HIGH", "Dynamic file inclusion risk"),
                # XSS
                (r"echo\s+\$.*;", "MEDIUM", "Direct variable output (potential XSS)"),
                (r"print\s+\$.*;", "MEDIUM", "Direct variable output (potential XSS)"),
                # Hardcoded secrets
                (r"\$password\s*=\s*['\"].{4,}['\"]", "HIGH", "Possible hardcoded password"),
                (r"\$secret\s*=\s*['\"].{4,}['\"]", "HIGH", "Possible hardcoded secret"),
                # Insecure randomness
                (r"rand\(", "MEDIUM", "Insecure random for cryptography"),
                (r"mt_rand\(", "MEDIUM", "Insecure random for security contexts"),
            ],

            "ruby": [
                # SQL Injection
                (r"execute\(.*\+.*\)", "HIGH", "Possible SQL injection (string concatenation)"),
                (r"ActiveRecord::Base\.connection\.execute\(.*\+.*\)", "HIGH", "Possible SQL injection"),
                # Command injection
                (r"`.*#\{.*\}.*`", "CRITICAL", "Command injection in backticks"),
                (r"system\(.*#\{.*\}.*\)", "CRITICAL", "Command injection in system()"),
                (r"exec\(.*#\{.*\}.*\)", "CRITICAL", "Command injection in exec()"),
                # Deserialization
                (r"Marshal\.load\(", "CRITICAL", "Unsafe deserialization"),
                (r"YAML\.load\(", "HIGH", "Unsafe YAML deserialization"),
                # Eval usage
                (r"eval\(", "CRITICAL", "Unsafe eval() usage"),
                (r"instance_eval\(", "HIGH", "Dynamic code evaluation"),
                # Hardcoded secrets
                (r"password\s*=\s*['\"].{4,}['\"]", "HIGH", "Possible hardcoded password"),
                (r"secret\s*=\s*['\"].{4,}['\"]", "HIGH", "Possible hardcoded secret"),
                (r"api_key\s*=\s*['\"].{8,}['\"]", "HIGH", "Possible hardcoded API key"),
                # Insecure randomness
                (r"rand\(", "MEDIUM", "Insecure random for cryptography"),
            ],

            "go": [
                # Command injection
                (r"exec\.CommandContext\(", "CRITICAL", "Command execution (validate inputs)"),
                (r"exec\.Command\(", "CRITICAL", "Command execution (validate inputs)"),
                # SQL Injection
                (r"db\.Exec\(.*\+.*\)", "HIGH", "Possible SQL injection (string concatenation)"),
                (r"db\.Query\(.*\+.*\)", "HIGH", "Possible SQL injection (string concatenation)"),
                # File path traversal
                (r"filepath\.Join\(.*\.\.", "MEDIUM", "Possible path traversal"),
                # Hardcoded secrets
                (r"password\s*:=\s*\".*\"", "HIGH", "Possible hardcoded password"),
                (r"secret\s*:=\s*\".*\"", "HIGH", "Possible hardcoded secret"),
                (r"apiKey\s*:=\s*\".*\"", "HIGH", "Possible hardcoded API key"),
                # Insecure randomness
                (r"math/rand", "MEDIUM", "Insecure random for cryptography"),
                # Memory safety
                (r"unsafe\.Pointer", "HIGH", "Unsafe memory operations"),
            ]
        }

        # Tool installation status
        self.tools_available = {}
        self._check_tools_availability()

    def _check_tools_availability(self):
        """Check which security tools are available on the system."""
        # Python - Bandit
        try:
            subprocess.run(["bandit", "--version"], capture_output=True, timeout=2, check=False)
            self.tools_available["bandit"] = True
        except:
            self.tools_available["bandit"] = False

        # JavaScript - ESLint with security plugin
        try:
            subprocess.run(["npx", "eslint", "--version"], capture_output=True, timeout=2, check=False)
            self.tools_available["eslint"] = True
        except:
            try:
                subprocess.run(["eslint", "--version"], capture_output=True, timeout=2, check=False)
                self.tools_available["eslint"] = True
            except:
                self.tools_available["eslint"] = False

        # Go - go vet
        try:
            subprocess.run(["go", "version"], capture_output=True, timeout=2, check=False)
            self.tools_available["go_vet"] = True
        except:
            self.tools_available["go_vet"] = False

        print(f"✅ Security scanner initialized. Tools available: {self.tools_available}")

    def _create_temp_file(self, code: str, extension: str) -> str:
        """Create a temporary file with the given code."""
        fd, path = tempfile.mkstemp(suffix=extension, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)
        except:
            os.close(fd)
            raise
        return path

    def _run_regex_scan(self, language: str, code: str, filename: str) -> List[Dict[str, Any]]:
        """Run regex-based vulnerability scanning."""
        findings = []
        lines = code.split('\n')

        if language not in self.vulnerability_patterns:
            return findings

        for pattern, severity, description in self.vulnerability_patterns[language]:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        # Extract the matching part
                        match = regex.search(line)
                        matched_text = match.group(0) if match else line.strip()

                        finding = {
                            "line": i,
                            "severity": severity,
                            "confidence": "HIGH",
                            "tool": "regex_scanner",
                            "rule_id": f"SEC_{severity}_{abs(hash(pattern)) % 10000:04d}",
                            "description": description,
                            "matched_text": matched_text[:100],
                            "category": self._categorize_finding(description),
                            "remediation": self._get_remediation(description)
                        }
                        findings.append(finding)
            except Exception as e:
                # Skip problematic patterns
                continue

        return findings

    def _categorize_finding(self, description: str) -> str:
        """Categorize security finding."""
        description_lower = description.lower()

        if any(word in description_lower for word in ["sql", "injection"]):
            return "SQL Injection"
        elif any(word in description_lower for word in ["command", "exec", "shell"]):
            return "Command Injection"
        elif any(word in description_lower for word in ["xss", "cross-site"]):
            return "Cross-Site Scripting (XSS)"
        elif any(word in description_lower for word in ["deserialization", "pickle", "yaml", "marshal"]):
            return "Insecure Deserialization"
        elif any(word in description_lower for word in ["eval", "function", "dynamic"]):
            return "Code Injection"
        elif any(word in description_lower for word in ["password", "secret", "key", "hardcoded"]):
            return "Hardcoded Credentials"
        elif any(word in description_lower for word in ["random", "crypto", "md5", "sha1", "hash"]):
            return "Cryptographic Issues"
        elif any(word in description_lower for word in ["ssl", "tls", "verify"]):
            return "SSL/TLS Issues"
        elif any(word in description_lower for word in ["path", "traversal", "include"]):
            return "Path Traversal"
        else:
            return "Security Issue"

    def _get_remediation(self, description: str) -> str:
        """Get remediation advice for a finding."""
        description_lower = description.lower()

        if "sql injection" in description_lower:
            return "Use parameterized queries or prepared statements instead of string concatenation."
        elif "command injection" in description_lower:
            return "Avoid shell=True, validate and sanitize all user inputs, use subprocess without shell."
        elif "xss" in description_lower:
            return "Use proper output encoding, escape special characters, or use templating engines with auto-escaping."
        elif "deserialization" in description_lower:
            return "Avoid untrusted deserialization, use safe alternatives (json.loads, yaml.safe_load)."
        elif "eval" in description_lower:
            return "Avoid eval() and Function() constructor. Use JSON.parse() for JSON or other safer alternatives."
        elif "hardcoded" in description_lower:
            return "Use environment variables or secure secret management systems (AWS Secrets Manager, HashiCorp Vault)."
        elif "md5" in description_lower or "sha1" in description_lower:
            return "Use stronger hashing algorithms (SHA-256, SHA-512, bcrypt, Argon2)."
        elif "random" in description_lower:
            return "For security contexts, use cryptographically secure random (secrets module in Python, crypto.randomBytes in JS)."
        elif "ssl" in description_lower:
            return "Always verify SSL certificates. Use proper certificate validation."
        elif "localstorage" in description_lower:
            return "Avoid storing sensitive data in localStorage. Use HTTP-only cookies or secure session storage."
        else:
            return "Review code for security implications and apply principle of least privilege."

    def scan(self, language: str, code: str, filename: str = "") -> Dict[str, Any]:
        """
        Run security scan for the given language and code.

        Args:
            language: Programming language
            code: Source code to scan
            filename: Original filename

        Returns:
            Security scan results
        """
        print(f"🔒 Starting security scan for {language} (file: {filename})")

        if language not in self.security_tools:
            return {
                "passed": True,
                "score": 100,
                "findings": [],
                "risk_level": "LOW",
                "tools_used": [],
                "error": f"Unsupported language for security scan: {language}"
            }

        try:
            # Run language-specific security scan
            scan_result = self.security_tools[language](code, filename)

            # Run regex-based scan
            regex_findings = self._run_regex_scan(language, code, filename)

            # Combine findings
            all_findings = scan_result.get("findings", []) + regex_findings

            # Remove duplicates (by line and description)
            unique_findings = []
            seen = set()
            for finding in all_findings:
                key = (finding.get("line", 0), finding.get("description", ""))
                if key not in seen:
                    seen.add(key)
                    unique_findings.append(finding)

            # Calculate score and risk level
            score, risk_level = self._calculate_security_score(unique_findings)

            # Prepare tools used list
            tools_used = scan_result.get("tools_used", [])
            if regex_findings:
                tools_used.append("regex_scanner")
            if not tools_used:
                tools_used = ["regex_scanner"]

            print(f"✅ Security scan completed: {len(unique_findings)} findings, score: {score}/100")

            return {
                "passed": risk_level in ["LOW", "INFO"],
                "score": score,
                "findings": unique_findings,
                "risk_level": risk_level,
                "tools_used": tools_used,
                "summary": self._generate_summary(unique_findings)
            }

        except Exception as e:
            print(f"❌ Security scan error: {str(e)}")
            return {
                "passed": False,
                "score": 0,
                "findings": [],
                "risk_level": "HIGH",
                "tools_used": [],
                "error": f"Security scan failed: {str(e)}"
            }

    def _calculate_security_score(self, findings: List[Dict[str, Any]]) -> Tuple[int, str]:
        """Calculate security score (0-100) and risk level."""
        if not findings:
            return 100, "LOW"

        # Weight findings by severity
        severity_weights = {
            "CRITICAL": 20,
            "HIGH": 10,
            "MEDIUM": 5,
            "LOW": 2,
            "INFO": 1
        }

        total_weight = 0
        for finding in findings:
            severity = finding.get("severity", "MEDIUM")
            total_weight += severity_weights.get(severity, 5)

        # Calculate score (inverse of weight)
        # Scale: 0-10 weight = 100-90 score, 10-30 weight = 90-70 score, 30+ = 70-0 score
        if total_weight <= 10:
            score = 100 - total_weight
        elif total_weight <= 30:
            score = 90 - ((total_weight - 10) * 1)
        else:
            score = max(70 - ((total_weight - 30) * 2), 0)

        score = max(0, min(100, int(score)))

        # Determine risk level based on highest severity
        severities = [f.get("severity", "MEDIUM") for f in findings]
        if "CRITICAL" in severities or total_weight >= 40:
            risk_level = "CRITICAL"
        elif "HIGH" in severities or total_weight >= 20:
            risk_level = "HIGH"
        elif "MEDIUM" in severities or total_weight >= 10:
            risk_level = "MEDIUM"
        elif "LOW" in severities or total_weight >= 5:
            risk_level = "LOW"
        else:
            risk_level = "INFO"

        return score, risk_level

    def _generate_summary(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of security findings."""
        if not findings:
            return {
                "total_findings": 0,
                "by_severity": {},
                "by_category": {},
                "recommendation": "No security issues found."
            }

        by_severity = {}
        by_category = {}

        for finding in findings:
            severity = finding.get("severity", "MEDIUM")
            category = finding.get("category", "Security Issue")

            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_category[category] = by_category.get(category, 0) + 1

        # Generate recommendation
        if by_severity.get("CRITICAL", 0) > 0:
            recommendation = "CRITICAL issues found. Immediate attention required!"
        elif by_severity.get("HIGH", 0) > 0:
            recommendation = "HIGH severity issues found. Address before deployment."
        elif by_severity.get("MEDIUM", 0) > 0:
            recommendation = "MEDIUM severity issues found. Review and fix."
        elif by_severity.get("LOW", 0) > 0:
            recommendation = "Low severity issues found. Consider fixing."
        else:
            recommendation = "Minor informational findings."

        return {
            "total_findings": len(findings),
            "by_severity": by_severity,
            "by_category": by_category,
            "recommendation": recommendation
        }

    # ========== LANGUAGE-SPECIFIC SCANNERS ==========

    def _scan_python(self, code: str, filename: str) -> Dict[str, Any]:
        """Python security scan using Bandit."""
        findings = []
        tools_used = []

        # Always run regex scan first
        regex_findings = self._run_regex_scan("python", code, filename)
        findings.extend(regex_findings)

        # Run Bandit (if available)
        if self.tools_available.get("bandit", False):
            try:
                temp_file = self._create_temp_file(code, ".py")

                try:
                    # Run bandit with JSON output
                    result = subprocess.run(
                        ["bandit", "-f", "json", "-ll", temp_file],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False
                    )

                    if result.returncode in [0, 1]:  # 0 = success, 1 = issues found
                        try:
                            bandit_results = json.loads(result.stdout)
                            for issue in bandit_results.get("results", []):
                                finding = {
                                    "line": issue.get("line_number", 1),
                                    "severity": issue.get("issue_severity", "MEDIUM").upper(),
                                    "confidence": issue.get("issue_confidence", "MEDIUM").upper(),
                                    "tool": "bandit",
                                    "rule_id": issue.get("test_id", "unknown"),
                                    "description": issue.get("issue_text", "Security issue"),
                                    "matched_text": issue.get("code", "")[:100],
                                    "category": self._categorize_finding(issue.get("issue_text", "")),
                                    "remediation": self._get_remediation(issue.get("issue_text", ""))
                                }
                                findings.append(finding)
                                print(f"   Bandit finding: Line {finding['line']} - {finding['description']}")
                        except json.JSONDecodeError:
                            print("   Could not parse Bandit JSON output")

                        tools_used.append("bandit")
                    else:
                        print(f"   Bandit failed with return code: {result.returncode}")

                except subprocess.TimeoutExpired:
                    print("   Bandit timed out")
                finally:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)

            except Exception as e:
                print(f"   Bandit scan failed: {e}")

        else:
            print("   Bandit not available, using regex scanner only")

        return {"findings": findings, "tools_used": tools_used}

    def _scan_javascript(self, code: str, filename: str) -> Dict[str, Any]:
        """JavaScript security scan using ESLint security rules."""
        findings = []
        tools_used = []

        # Always run regex scan
        regex_findings = self._run_regex_scan("javascript", code, filename)
        findings.extend(regex_findings)

        # Run ESLint with security focus (if available)
        if self.tools_available.get("eslint", False):
            try:
                temp_file = self._create_temp_file(code, ".js")

                try:
                    # Create ESLint command
                    cmd = ["npx", "eslint", "--no-eslintrc", "--format", "json", temp_file]

                    # Add rule configurations
                    env = os.environ.copy()
                    env['ESLINT_OPTIONS'] = '{"rules":{"no-eval":"error","no-implied-eval":"error"}}'

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=20,
                        env=env,
                        check=False
                    )

                    if result.returncode in [0, 1]:
                        try:
                            eslint_results = json.loads(result.stdout)
                            if isinstance(eslint_results, list):
                                for file_result in eslint_results:
                                    for message in file_result.get("messages", []):
                                        if message.get("severity") in [1, 2]:
                                            severity = "HIGH" if message.get("severity") == 2 else "MEDIUM"
                                            if "eval" in str(message.get("message", "")).lower():
                                                severity = "CRITICAL"

                                            finding = {
                                                "line": message.get("line", 1),
                                                "severity": severity,
                                                "confidence": "HIGH",
                                                "tool": "eslint",
                                                "rule_id": message.get("ruleId", "security-rule"),
                                                "description": message.get("message", "Security issue"),
                                                "matched_text": "",
                                                "category": self._categorize_finding(message.get("message", "")),
                                                "remediation": self._get_remediation(message.get("message", ""))
                                            }
                                            findings.append(finding)
                                            print(f"   ESLint finding: Line {finding['line']} - {finding['description']}")
                        except json.JSONDecodeError:
                            # Couldn't parse JSON output, check for text output
                            if "eval" in result.stdout.lower() or "security" in result.stdout.lower():
                                for line in result.stdout.split('\n'):
                                    if "error" in line.lower() and "eval" in line.lower():
                                        finding = {
                                            "line": 1,
                                            "severity": "CRITICAL",
                                            "confidence": "MEDIUM",
                                            "tool": "eslint",
                                            "rule_id": "no-eval",
                                            "description": "eval() usage detected",
                                            "matched_text": "",
                                            "category": "Code Injection",
                                            "remediation": self._get_remediation("eval")
                                        }
                                        findings.append(finding)

                        tools_used.append("eslint")
                finally:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)

            except Exception as e:
                print(f"   ESLint scan failed: {e}")

        return {"findings": findings, "tools_used": tools_used}

    def _scan_java(self, code: str, filename: str) -> Dict[str, Any]:
        """Java security scan using regex patterns."""
        findings = []
        tools_used = ["regex_scanner"]

        # Run regex scan
        regex_findings = self._run_regex_scan("java", code, filename)
        findings.extend(regex_findings)

        # Additional Java-specific patterns
        java_specific_patterns = [
            (r"ProcessBuilder\(", "HIGH", "ProcessBuilder usage (validate inputs)"),
            (r"SecurityManager", "INFO", "SecurityManager usage (deprecated in newer Java)"),
            (r"AccessController\.doPrivileged", "HIGH", "Privileged block (review permissions)"),
            (r"//\s*TODO.*security", "LOW", "Security TODO comment"),
            (r"//\s*FIXME.*security", "LOW", "Security FIXME comment"),
            (r"\.getConnection\(.*\)", "MEDIUM", "Database connection - check for proper configuration"),
        ]

        lines = code.split('\n')
        for pattern, severity, description in java_specific_patterns:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        finding = {
                            "line": i,
                            "severity": severity,
                            "confidence": "MEDIUM",
                            "tool": "regex_scanner",
                            "rule_id": f"JAVA_{severity}_{abs(hash(pattern)) % 10000:04d}",
                            "description": description,
                            "matched_text": line.strip()[:100],
                            "category": self._categorize_finding(description),
                            "remediation": self._get_remediation(description)
                        }
                        findings.append(finding)
            except:
                continue

        print(f"   Java scan: {len(findings)} findings")
        return {"findings": findings, "tools_used": tools_used}

    def _scan_php(self, code: str, filename: str) -> Dict[str, Any]:
        """PHP security scan."""
        findings = []
        tools_used = ["regex_scanner"]

        # Run regex scan
        regex_findings = self._run_regex_scan("php", code, filename)
        findings.extend(regex_findings)

        # PHP-specific patterns
        php_specific_patterns = [
            (r"extract\(\$_(GET|POST|REQUEST|COOKIE|SERVER)\)", "CRITICAL", "Dangerous extract() on superglobal"),
            (r"phpinfo\(", "MEDIUM", "phpinfo() exposure in production"),
            (r"display_errors\s*=\s*1", "MEDIUM", "Error display enabled in production"),
            (r"error_reporting\s*\(\s*E_ALL\s*\)", "MEDIUM", "Full error reporting in production"),
            (r"\$_(GET|POST|REQUEST)\[.*\].*header\(", "HIGH", "Header injection vulnerability"),
        ]

        lines = code.split('\n')
        for pattern, severity, description in php_specific_patterns:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        finding = {
                            "line": i,
                            "severity": severity,
                            "confidence": "HIGH",
                            "tool": "regex_scanner",
                            "rule_id": f"PHP_{severity}_{abs(hash(pattern)) % 10000:04d}",
                            "description": description,
                            "matched_text": line.strip()[:100],
                            "category": self._categorize_finding(description),
                            "remediation": self._get_remediation(description)
                        }
                        findings.append(finding)
            except:
                continue

        print(f"   PHP scan: {len(findings)} findings")
        return {"findings": findings, "tools_used": tools_used}

    def _scan_ruby(self, code: str, filename: str) -> Dict[str, Any]:
        """Ruby security scan."""
        findings = []
        tools_used = ["regex_scanner"]

        # Run regex scan
        regex_findings = self._run_regex_scan("ruby", code, filename)
        findings.extend(regex_findings)

        # Ruby-specific patterns
        ruby_specific_patterns = [
            (r"Kernel\.open\(", "CRITICAL", "Kernel.open() with user input (command injection risk)"),
            (r"open\s*\(\s*['\"]\|", "CRITICAL", "open() with pipe (command injection)"),
            (r"send\s*\(\s*[:\"].*['\"]\s*,\s*.*\)", "HIGH", "Dynamic method send() with user input"),
            (r"constantize\(", "HIGH", "constantize() with user input (class injection)"),
            (r"eval\s*\(.*\)", "CRITICAL", "eval() with dynamic input"),
            (r"instance_eval\s*\(.*\)", "HIGH", "instance_eval() with dynamic input"),
        ]

        lines = code.split('\n')
        for pattern, severity, description in ruby_specific_patterns:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        finding = {
                            "line": i,
                            "severity": severity,
                            "confidence": "HIGH",
                            "tool": "regex_scanner",
                            "rule_id": f"RUBY_{severity}_{abs(hash(pattern)) % 10000:04d}",
                            "description": description,
                            "matched_text": line.strip()[:100],
                            "category": self._categorize_finding(description),
                            "remediation": self._get_remediation(description)
                        }
                        findings.append(finding)
            except:
                continue

        print(f"   Ruby scan: {len(findings)} findings")
        return {"findings": findings, "tools_used": tools_used}

    def _scan_go(self, code: str, filename: str) -> Dict[str, Any]:
        """Go security scan."""
        findings = []
        tools_used = []

        # Run regex scan
        regex_findings = self._run_regex_scan("go", code, filename)
        findings.extend(regex_findings)

        # Run go vet (if available)
        if self.tools_available.get("go_vet", False):
            try:
                # Create a temporary Go module
                temp_dir = tempfile.mkdtemp()
                go_mod_path = os.path.join(temp_dir, "go.mod")
                go_file_path = os.path.join(temp_dir, "main.go")

                # Create go.mod
                with open(go_mod_path, "w") as f:
                    f.write("module temp\ngo 1.19\n")

                # Write Go code
                with open(go_file_path, "w") as f:
                    f.write(code)

                try:
                    # Run go vet
                    result = subprocess.run(
                        ["go", "vet", "./..."],
                        capture_output=True,
                        text=True,
                        cwd=temp_dir,
                        timeout=20,
                        check=False
                    )

                    if result.stderr:
                        # Parse go vet output
                        for line in result.stderr.split('\n'):
                            if line.strip() and ":" in line:
                                # Format: filename:line:col: message
                                parts = line.split(":", 3)
                                if len(parts) >= 4:
                                    try:
                                        line_num = int(parts[1])
                                        message = parts[3].strip()

                                        # Classify severity
                                        severity = "MEDIUM"
                                        if any(word in message.lower() for word in ["insecure", "unsafe", "injection"]):
                                            severity = "HIGH"
                                        elif any(word in message.lower() for word in ["error", "fatal", "panic"]):
                                            severity = "CRITICAL"

                                        finding = {
                                            "line": line_num,
                                            "severity": severity,
                                            "confidence": "MEDIUM",
                                            "tool": "go_vet",
                                            "rule_id": "GO_VET",
                                            "description": message,
                                            "matched_text": "",
                                            "category": self._categorize_finding(message),
                                            "remediation": self._get_remediation(message)
                                        }
                                        findings.append(finding)
                                        print(f"   go vet finding: Line {finding['line']} - {finding['description']}")
                                    except (ValueError, IndexError):
                                        continue

                        tools_used.append("go_vet")
                finally:
                    # Clean up temp directory
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)

            except Exception as e:
                print(f"   go vet scan failed: {e}")

        # Go-specific regex patterns
        go_specific_patterns = [
            (r"ioutil\.ReadFile\(", "MEDIUM", "ioutil.ReadFile() is deprecated (use os.ReadFile)"),
            (r"ioutil\.WriteFile\(", "MEDIUM", "ioutil.WriteFile() is deprecated (use os.WriteFile)"),
            (r"//\s*TODO.*security", "LOW", "Security TODO comment"),
            (r"//\s*FIXME.*security", "LOW", "Security FIXME comment"),
            (r"crypto/md5", "MEDIUM", "MD5 usage (cryptographically weak)"),
            (r"crypto/sha1", "MEDIUM", "SHA-1 usage (cryptographically weak)"),
        ]

        lines = code.split('\n')
        for pattern, severity, description in go_specific_patterns:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        finding = {
                            "line": i,
                            "severity": severity,
                            "confidence": "MEDIUM",
                            "tool": "regex_scanner",
                            "rule_id": f"GO_{severity}_{abs(hash(pattern)) % 10000:04d}",
                            "description": description,
                            "matched_text": line.strip()[:100],
                            "category": self._categorize_finding(description),
                            "remediation": self._get_remediation(description)
                        }
                        findings.append(finding)
            except:
                continue

        if not tools_used:
            tools_used.append("regex_scanner")

        print(f"   Go scan: {len(findings)} findings")
        return {"findings": findings, "tools_used": tools_used}

    def generate_security_report(self, scan_results: Dict[str, Any]) -> str:
        """Generate a human-readable security report."""
        report = []

        report.append("🔒 SECURITY SCAN REPORT")
        report.append("=" * 70)

        # Overall status
        passed = scan_results.get("passed", False)
        score = scan_results.get("score", 0)
        risk_level = scan_results.get("risk_level", "UNKNOWN")
        tools_used = scan_results.get("tools_used", [])

        report.append(f"\n📊 OVERALL STATUS:")
        report.append(f"  Security Score: {score}/100")
        report.append(f"  Risk Level: {risk_level}")
        report.append(f"  Status: {'✅ PASSED' if passed else '❌ SECURITY ISSUES DETECTED'}")
        report.append(f"  Tools Used: {', '.join(tools_used) if tools_used else 'Regex scanner only'}")

        # Summary
        summary = scan_results.get("summary", {})
        if summary:
            report.append(f"\n📈 SUMMARY:")
            report.append(f"  Total Findings: {summary.get('total_findings', 0)}")

            by_severity = summary.get("by_severity", {})
            if by_severity:
                report.append(f"  Findings by Severity:")
                for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
                    if severity in by_severity:
                        report.append(f"    {severity}: {by_severity[severity]}")

            recommendation = summary.get("recommendation", "")
            if recommendation:
                report.append(f"\n💡 RECOMMENDATION: {recommendation}")

        # Detailed findings
        findings = scan_results.get("findings", [])
        if findings:
            # Group by severity
            grouped_findings = {}
            for finding in findings:
                severity = finding.get("severity", "MEDIUM")
                if severity not in grouped_findings:
                    grouped_findings[severity] = []
                grouped_findings[severity].append(finding)

            # Show findings by severity (highest first)
            for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
                if severity in grouped_findings:
                    report.append(f"\n{'⚠️ ' if severity in ['CRITICAL', 'HIGH'] else '🔸 '}{severity} SEVERITY FINDINGS:")

                    for i, finding in enumerate(grouped_findings[severity][:10], 1):  # Limit to 10 per severity
                        line = finding.get("line", "?")
                        description = finding.get("description", "")
                        category = finding.get("category", "")
                        tool = finding.get("tool", "unknown")

                        report.append(f"  {i}. [Line {line}] {description}")
                        report.append(f"     Category: {category} | Tool: {tool}")

                        remediation = finding.get("remediation", "")
                        if remediation:
                            report.append(f"     💡 Fix: {remediation}")

                    if len(grouped_findings[severity]) > 10:
                        report.append(f"     ... and {len(grouped_findings[severity]) - 10} more")

        else:
            report.append(f"\n🎉 No security findings detected!")

        # Error if any
        error = scan_results.get("error")
        if error:
            report.append(f"\n⚠️  SCAN ERROR: {error}")

        report.append("\n" + "=" * 70)
        report.append("🔍 Scan completed. Review findings above.")

        return "\n".join(report)


# Singleton instance
_security_scanner_instance = None

def get_security_scanner() -> SecurityScanner:
    """Get singleton instance of SecurityScanner."""
    global _security_scanner_instance
    if _security_scanner_instance is None:
        _security_scanner_instance = SecurityScanner()
    return _security_scanner_instance


# Test function
def test_security_scanner():
    """Test the security scanner with sample code."""
    scanner = SecurityScanner()

    print("🔒 SECURITY SCANNER TEST")
    print("=" * 70)

    # Test vulnerable Python code
    vulnerable_python = """
import os
import subprocess
import pickle

def unsafe_function(user_input):
    # SQL Injection vulnerability
    query = "SELECT * FROM users WHERE id = " + user_input
    cursor.execute(query)

    # Command injection vulnerability
    os.system("echo " + user_input)

    # Unsafe deserialization
    data = pickle.loads(user_input)

    return data
"""

    print("\n🧪 Testing Python security scan...")
    result = scanner.scan("python", vulnerable_python, "vulnerable.py")
    print(f"Score: {result['score']}/100")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Findings: {len(result['findings'])}")
    for finding in result['findings']:
        print(f"  [{finding['severity']}] Line {finding['line']}: {finding['description']}")

    # Test vulnerable JavaScript code
    vulnerable_js = """
function unsafeFunction(userInput) {
    // XSS vulnerability
    document.getElementById('output').innerHTML = userInput;

    // Eval vulnerability
    eval(userInput);

    // Command injection-like
    const result = "ls " + userInput;

    return result;
}
"""

    print("\n🧪 Testing JavaScript security scan...")
    result = scanner.scan("javascript", vulnerable_js, "vulnerable.js")
    print(f"Score: {result['score']}/100")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Findings: {len(result['findings'])}")
    for finding in result['findings']:
        print(f"  [{finding['severity']}] Line {finding['line']}: {finding['description']}")

    # Test vulnerable Java code
    vulnerable_java = """
public class Vulnerable {
    private static final String PASSWORD = "secret123";

    public void unsafeMethod(String input) {
        // SQL Injection
        String query = "SELECT * FROM users WHERE id = " + input;

        // Command injection
        Runtime.getRuntime().exec("echo " + input);
    }
}
"""

    print("\n🧪 Testing Java security scan...")
    result = scanner.scan("java", vulnerable_java, "Vulnerable.java")
    print(f"Score: {result['score']}/100")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Findings: {len(result['findings'])}")
    for finding in result['findings']:
        print(f"  [{finding['severity']}] Line {finding['line']}: {finding['description']}")


if __name__ == "__main__":
    test_security_scanner()
