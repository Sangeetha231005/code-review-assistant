"""
Linter Runner Module
Runs appropriate linters for each programming language
Supports only 6 languages: Python, Java, JavaScript, PHP, Ruby, Go
"""
import subprocess
import tempfile
import os
import re
import json
import shutil
from typing import Dict, List, Any, Optional, Tuple

class LinterRunner:
    """Runs language-specific linters and returns results."""
    def __init__(self):
        self.linter_commands = {
            "python": self._run_python_lint,
            "java": self._run_java_lint,
            "javascript": self._run_javascript_lint,
            "php": self._run_php_lint,
            "ruby": self._run_ruby_lint,
            "go": self._run_go_lint
        }

        # Linter configuration
        self.config = {
            "max_line_length": 100,
            "max_complexity": 10,
            "timeout_seconds": 30
        }

    def run_linter(self, language: str, code: str, filename: str) -> Dict[str, Any]:
        """
        Run appropriate linter for the given language.

        Args:
            language: Programming language
            code: Source code to analyze
            filename: Original filename (for context)

        Returns:
            Dictionary with linting results
        """
        if language not in self.linter_commands:
            return self._create_error_result(f"Unsupported language: {language}")

        try:
            return self.linter_commands[language](code, filename)
        except subprocess.TimeoutExpired:
            return self._create_error_result("Linter timeout - code too complex")
        except Exception as e:
            return self._create_error_result(f"Linter failed: {str(e)}")

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

    def _create_error_result(self, message: str) -> Dict[str, Any]:
        """Create an error result dictionary."""
        return {
            "passed": False,
            "score": 0,
            "errors": [{"line": 1, "message": message, "severity": "error", "tool": "system"}],
            "warnings": [],
            "info": [],
            "raw_output": "",
            "metrics": {
                "lines_of_code": 0,
                "complexity": 0,
                "maintainability": 0
            }
        }

    def _calculate_score(self, errors: List, warnings: List, info: List) -> int:
        """Calculate a quality score (0-100)."""
        base_score = 100

        # Penalties
        error_penalty = min(len(errors) * 15, 70)  # Max 70 penalty
        warning_penalty = min(len(warnings) * 5, 25)  # Max 25 penalty
        info_penalty = min(len(info) * 2, 5)  # Max 5 penalty

        score = max(0, base_score - error_penalty - warning_penalty - info_penalty)

        # Bonus for no issues
        if len(errors) == 0 and len(warnings) == 0:
            score = min(100, score + 10)

        return score

    def _parse_pylint_output(self, output: str) -> Tuple[List, List, List]:
        """Parse pylint output into errors, warnings, and info."""
        errors = []
        warnings = []
        info = []

        for line in output.split("\n"):
            if not line.strip():
                continue

            # Parse pylint format: filename:line:column: message-id: message
            match = re.match(r"^.*?:(\d+):(\d+):\s*([A-Z]\d+):\s*(.+)$", line)
            if match:
                line_num = int(match.group(1))
                message_id = match.group(3)
                message = match.group(4)

                item = {
                    "line": line_num,
                    "message": message,
                    "severity": "",
                    "tool": "pylint",
                    "code": message_id
                }

                if message_id.startswith("E") or message_id.startswith("F"):
                    item["severity"] = "error"
                    errors.append(item)
                elif message_id.startswith("W"):
                    item["severity"] = "warning"
                    warnings.append(item)
                elif message_id.startswith("C") or message_id.startswith("R"):
                    item["severity"] = "info"
                    info.append(item)

        return errors, warnings, info

    def _run_python_lint(self, code: str, filename: str) -> Dict[str, Any]:
        """Run Python linters (pylint, flake8)."""
        temp_file = self._create_temp_file(code, ".py")
        all_errors = []
        all_warnings = []
        all_info = []

        try:
            # 1. Run pylint
            try:
                result = subprocess.run(
                    ["pylint", "--output-format=text", "--score=no", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.config["timeout_seconds"]
                )

                if result.returncode in [0, 4, 8, 16, 28]:  # Valid pylint exit codes
                    errors, warnings, info = self._parse_pylint_output(result.stdout)
                    all_errors.extend(errors)
                    all_warnings.extend(warnings)
                    all_info.extend(info)
            except Exception as e:
                all_warnings.append({
                    "line": 1,
                    "message": f"pylint failed: {str(e)}",
                    "severity": "warning",
                    "tool": "pylint"
                })

            # 2. Run flake8
            try:
                result = subprocess.run(
                    ["flake8", "--max-line-length=100", "--format=default", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                for line in result.stdout.split("\n"):
                    if not line.strip():
                        continue

                    # Parse flake8 format: filename:line:col: code message
                    parts = line.split(":")
                    if len(parts) >= 4:
                        line_num = int(parts[1])
                        code_part = parts[3].strip().split(" ", 1)[0]
                        message = parts[3].strip().split(" ", 1)[1] if " " in parts[3].strip() else parts[3].strip()

                        severity = "warning"
                        if any(c in code_part for c in ["E9", "F4", "F6", "F7", "F8", "F82"]):
                            severity = "error"

                        item = {
                            "line": line_num,
                            "message": message,
                            "severity": severity,
                            "tool": "flake8",
                            "code": code_part
                        }

                        if severity == "error":
                            all_errors.append(item)
                        else:
                            all_warnings.append(item)
            except Exception as e:
                all_warnings.append({
                    "line": 1,
                    "message": f"flake8 failed: {str(e)}",
                    "severity": "warning",
                    "tool": "flake8"
                })

            # Calculate metrics
            lines = code.split("\n")
            loc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])

            score = self._calculate_score(all_errors, all_warnings, all_info)

            return {
                "passed": len(all_errors) == 0,
                "score": score,
                "errors": all_errors[:50],
                "warnings": all_warnings[:50],
                "info": all_info[:20],
                "raw_output": "",
                "metrics": {
                    "lines_of_code": loc,
                    "complexity": len(all_errors) + len(all_warnings),
                    "maintainability": score
                }
            }

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _run_javascript_lint(self, code: str, filename: str) -> Dict[str, Any]:
        """Run JavaScript linters (eslint, jshint)."""
        temp_file = self._create_temp_file(code, ".js")
        all_errors = []
        all_warnings = []

        try:
            # 1. Run ESLint
            try:
                result = subprocess.run(
                    ["eslint", "--format=json", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=15
                )

                if result.returncode in [0, 1]:  # 0=no issues, 1=issues found
                    try:
                        eslint_results = json.loads(result.stdout)
                        for file_result in eslint_results:
                            for message in file_result.get("messages", []):
                                item = {
                                    "line": message.get("line", 1),
                                    "column": message.get("column", 1),
                                    "message": message.get("message", ""),
                                    "severity": message.get("severity", 1),
                                    "tool": "eslint",
                                    "rule": message.get("ruleId", "")
                                }

                                if message.get("severity") == 2:
                                    item["severity"] = "error"
                                    all_errors.append(item)
                                else:
                                    item["severity"] = "warning"
                                    all_warnings.append(item)
                    except json.JSONDecodeError:
                        # Parse text output
                        for line in result.stdout.split("\n"):
                            if "error" in line.lower() or "warning" in line.lower():
                                all_warnings.append({
                                    "line": 1,
                                    "message": line[:200],
                                    "severity": "warning",
                                    "tool": "eslint"
                                })
            except Exception as e:
                all_warnings.append({
                    "line": 1,
                    "message": f"ESLint failed: {str(e)}",
                    "severity": "warning",
                    "tool": "eslint"
                })

            # 2. Run JSHint
            try:
                result = subprocess.run(
                    ["jshint", "--reporter=json", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.stdout:
                    try:
                        jshint_results = json.loads(result.stdout)
                        for file_result in jshint_results:
                            for error in file_result.get("errors", []):
                                if error.get("code", "").startswith("E"):
                                    severity = "error"
                                    all_errors.append({
                                        "line": error.get("line", 1),
                                        "message": error.get("reason", ""),
                                        "severity": severity,
                                        "tool": "jshint"
                                    })
                                else:
                                    severity = "warning"
                                    all_warnings.append({
                                        "line": error.get("line", 1),
                                        "message": error.get("reason", ""),
                                        "severity": severity,
                                        "tool": "jshint"
                                    })
                    except:
                        pass
            except:
                pass

            # 3. Manual checks
            lines = code.split("\n")
            for i, line in enumerate(lines, 1):
                # Check for var usage
                if re.search(r"\bvar\s+\w+", line) and "//" not in line.split("var")[0]:
                    all_warnings.append({
                        "line": i,
                        "message": "Use 'let' or 'const' instead of 'var'",
                        "severity": "warning",
                        "tool": "manual"
                    })

                # Check for == instead of ===
                if (" == " in line or " != " in line) and "//" not in line:
                    all_warnings.append({
                        "line": i,
                        "message": "Use '===' or '!==' for strict equality",
                        "severity": "warning",
                        "tool": "manual"
                    })

                # Check line length
                if len(line) > self.config["max_line_length"]:
                    all_warnings.append({
                        "line": i,
                        "message": f"Line too long ({len(line)} > {self.config['max_line_length']})",
                        "severity": "warning",
                        "tool": "manual"
                    })

            # Calculate score
            loc = len([l for l in lines if l.strip() and not l.strip().startswith("//")])
            score = self._calculate_score(all_errors, all_warnings, [])

            return {
                "passed": len(all_errors) == 0,
                "score": score,
                "errors": all_errors[:50],
                "warnings": all_warnings[:50],
                "info": [],
                "raw_output": "",
                "metrics": {
                    "lines_of_code": loc,
                    "complexity": len(all_errors) + len(all_warnings),
                    "maintainability": score
                }
            }

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _run_java_lint(self, code: str, filename: str) -> Dict[str, Any]:
        """Run Java linter (javac with -Xlint)."""
        # Create temp directory for Java files
        temp_dir = tempfile.mkdtemp()

        # Extract class name from code or use default
        class_name = "TempClass"
        match = re.search(r'class\s+(\w+)', code)
        if match:
            class_name = match.group(1)

        # Create Java file with proper name
        java_file = os.path.join(temp_dir, f"{class_name}.java")

        try:
            # Write code to file
            with open(java_file, 'w', encoding='utf-8') as f:
                f.write(code)

            all_errors = []
            all_warnings = []

            # Compile with warnings
            result = subprocess.run(
                ['javac', java_file],
                capture_output=True,
                text=True,
                cwd=temp_dir,
                timeout=20
            )

            if result.returncode != 0:
                # Compilation failed
                error_msg = result.stderr.split('\n')[0] if result.stderr else "Compilation failed"
                all_errors.append({
                    'line': 1,
                    'message': error_msg[:200],
                    'severity': 'error',
                    'tool': 'javac'
                })
            else:
                # Compilation succeeded - check for warnings
                if result.stderr:
                    for line in result.stderr.split('\n'):
                        if 'warning:' in line.lower():
                            all_warnings.append({
                                'line': 1,
                                'message': line.strip()[:200],
                                'severity': 'warning',
                                'tool': 'javac'
                            })

            # Manual style checks
            lines = code.split('\n')
            for i, line in enumerate(lines, 1):
                # Check line length
                if len(line) > 120:
                    all_warnings.append({
                        'line': i,
                        'message': f'Line too long ({len(line)} > 120)',
                        'severity': 'warning',
                        'tool': 'manual'
                    })

                # Check for wildcard imports
                if line.strip().startswith('import') and '.*;' in line:
                    all_warnings.append({
                        'line': i,
                        'message': 'Avoid wildcard imports',
                        'severity': 'warning',
                        'tool': 'manual'
                    })

            # Calculate metrics
            loc = len([l for l in lines if l.strip() and not l.strip().startswith('//')])
            score = self._calculate_score(all_errors, all_warnings, [])

            # Clean up compiled class files if they exist
            class_file = os.path.join(temp_dir, f"{class_name}.class")
            if os.path.exists(class_file):
                os.unlink(class_file)

            return {
                'passed': len(all_errors) == 0,
                'score': score,
                'errors': all_errors[:50],
                'warnings': all_warnings[:50],
                'info': [],
                'raw_output': result.stderr[:1000] if result.stderr else '',
                'metrics': {
                    'lines_of_code': loc,
                    'complexity': len(all_errors) + len(all_warnings),
                    'maintainability': score
                }
            }

        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_php_lint(self, code: str, filename: str) -> Dict[str, Any]:
        """Run PHP linter (php -l)."""
        temp_file = self._create_temp_file(code, ".php")
        all_errors = []
        all_warnings = []

        try:
            # Syntax check
            result = subprocess.run(
                ["php", "-l", temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                # Parse PHP syntax error
                for line in result.stderr.split("\n"):
                    if "Parse error:" in line:
                        match = re.search(r"line (\d+)", line)
                        line_num = int(match.group(1)) if match else 1
                        all_errors.append({
                            "line": line_num,
                            "message": line.split("Parse error:")[-1].strip(),
                            "severity": "error",
                            "tool": "php"
                        })

            # Manual style checks
            lines = code.split("\n")
            for i, line in enumerate(lines, 1):
                # Check for short open tags
                if "<? " in line and not "<?php" in line:
                    all_warnings.append({
                        "line": i,
                        "message": "Use full PHP opening tag <?php instead of <?",
                        "severity": "warning",
                        "tool": "manual"
                    })

                # Check line length
                if len(line) > 120:
                    all_warnings.append({
                        "line": i,
                        "message": f"Line too long ({len(line)} > 120)",
                        "severity": "warning",
                        "tool": "manual"
                    })

            # Calculate metrics
            in_php_block = False
            loc = 0
            for line in lines:
                stripped = line.strip()
                if "<?php" in line or "<?=" in line:
                    in_php_block = True
                if "?>" in line:
                    in_php_block = False
                if in_php_block and stripped and not stripped.startswith("//"):
                    loc += 1

            score = self._calculate_score(all_errors, all_warnings, [])

            return {
                "passed": len(all_errors) == 0,
                "score": score,
                "errors": all_errors[:50],
                "warnings": all_warnings[:50],
                "info": [],
                "raw_output": result.stderr[:1000] if result.stderr else result.stdout[:1000],
                "metrics": {
                    "lines_of_code": loc,
                    "complexity": len(all_errors) + len(all_warnings),
                    "maintainability": score
                }
            }

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _run_ruby_lint(self, code: str, filename: str) -> Dict[str, Any]:
        """Run Ruby linter (ruby -c)."""
        temp_file = self._create_temp_file(code, ".rb")
        all_errors = []
        all_warnings = []

        try:
            # Syntax check
            result = subprocess.run(
                ["ruby", "-c", temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                # Parse Ruby syntax error
                for line in result.stderr.split("\n"):
                    match = re.search(r":(\d+):", line)
                    if match:
                        all_errors.append({
                            "line": int(match.group(1)),
                            "message": line.strip(),
                            "severity": "error",
                            "tool": "ruby"
                        })

            # Manual style checks
            lines = code.split("\n")
            for i, line in enumerate(lines, 1):
                # Check for then in if statements
                if line.strip().startswith("if ") and "then" in line:
                    all_warnings.append({
                        "line": i,
                        "message": "Avoid using 'then' in if statements",
                        "severity": "warning",
                        "tool": "manual"
                    })

                # Check line length
                if len(line) > 100:
                    all_warnings.append({
                        "line": i,
                        "message": f"Line too long ({len(line)} > 100)",
                        "severity": "warning",
                        "tool": "manual"
                    })

            # Calculate metrics
            loc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
            score = self._calculate_score(all_errors, all_warnings, [])

            return {
                "passed": len(all_errors) == 0,
                "score": score,
                "errors": all_errors[:50],
                "warnings": all_warnings[:50],
                "info": [],
                "raw_output": result.stderr[:1000] if result.stderr else "",
                "metrics": {
                    "lines_of_code": loc,
                    "complexity": len(all_errors) + len(all_warnings),
                    "maintainability": score
                }
            }

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def _run_go_lint(self, code: str, filename: str) -> Dict[str, Any]:
        """Run Go linter (gofmt, go build)."""
        temp_file = self._create_temp_file(code, ".go")
        all_errors = []
        all_warnings = []

        try:
            # Check gofmt formatting
            result = subprocess.run(
                ["gofmt", "-l", temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.stdout.strip():
                all_warnings.append({
                    "line": 1,
                    "message": "Code not formatted with gofmt",
                    "severity": "warning",
                    "tool": "gofmt"
                })

            # Try to build
            temp_dir = os.path.dirname(temp_file)
            temp_binary = os.path.join(temp_dir, "test_binary")

            result = subprocess.run(
                ["go", "build", "-o", temp_binary, temp_file],
                capture_output=True,
                text=True,
                timeout=20
            )

            if result.returncode != 0:
                for line in result.stderr.split("\n"):
                    match = re.search(r":(\d+):", line)
                    if match:
                        line_num = int(match.group(1))
                        message = line.split(":", 2)[-1].strip() if ":" in line else line.strip()

                        # Classify as error or warning
                        severity = "error"
                        if any(warn in line.lower() for warn in ["warning", "unused", "not used"]):
                            severity = "warning"

                        item = {
                            "line": line_num,
                            "message": message,
                            "severity": severity,
                            "tool": "go"
                        }

                        if severity == "error":
                            all_errors.append(item)
                        else:
                            all_warnings.append(item)

            # Manual style checks
            lines = code.split("\n")
            for i, line in enumerate(lines, 1):
                # Check line length
                if len(line) > 100:
                    all_warnings.append({
                        "line": i,
                        "message": f"Line too long ({len(line)} > 100)",
                        "severity": "warning",
                        "tool": "manual"
                    })

            # Calculate metrics
            loc = len([l for l in lines if l.strip() and not l.strip().startswith("//")])
            score = self._calculate_score(all_errors, all_warnings, [])

            return {
                "passed": len(all_errors) == 0,
                "score": score,
                "errors": all_errors[:50],
                "warnings": all_warnings[:50],
                "info": [],
                "raw_output": result.stderr[:1000] if result.stderr else "",
                "metrics": {
                    "lines_of_code": loc,
                    "complexity": len(all_errors) + len(all_warnings),
                    "maintainability": score
                }
            }

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists(temp_binary):
                os.unlink(temp_binary)


# Singleton instance
_linter_instance = None

def get_linter_runner() -> LinterRunner:
    """Get singleton instance of LinterRunner."""
    global _linter_instance
    if _linter_instance is None:
        _linter_instance = LinterRunner()
    return _linter_instance

if __name__ == "__main__":
    # Test the linter runner
    runner = LinterRunner()

    test_code = """
def test():
x = 5
return x
"""
    result = runner.run_linter("python", test_code, "test.py")
    print(f"Python lint result: {result['passed']}, Score: {result['score']}")
