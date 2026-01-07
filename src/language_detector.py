# PASTE THE FULL CODE ABOVE HERE
"""
Language Detector Module
Detects programming language from filename and code content
Supports only 6 languages: Python, Java, JavaScript, PHP, Ruby, Go
"""
import re
import os
from typing import Dict, Tuple, Optional, List

class LanguageDetector:
    """Detects programming language from file extension and code content."""
    def __init__(self):
        # Language extensions mapping (6 languages only)
        self.language_extensions = {
            'python': ['.py', '.pyw', '.pyi'],
            'java': ['.java'],
            'javascript': ['.js', '.jsx', '.mjs', '.cjs'],
            'php': ['.php', '.php3', '.php4', '.php5', '.phtml'],
            'ruby': ['.rb', '.rbw', '.rake', '.gemspec', '.ru'],
            'go': ['.go']
        }

        # Reverse mapping for quick lookup
        self.extension_to_language = {}
        for lang, exts in self.language_extensions.items():
            for ext in exts:
                self.extension_to_language[ext] = lang

        # Content patterns for language detection (6 languages only)
        self.content_patterns = {
            'python': [
                (r'^\s*(import\s+|from\s+\S+\s+import)', 3),
                (r'^\s*def\s+\w+\s*\(', 2),
                (r'^\s*class\s+\w+', 2),
                (r'^\s*@\w+', 1),
                (r'\bprint\(', 1),
                (r'\blambda\b', 1),
                (r'\bself\b', 1),
            ],
            'java': [
                (r'^\s*(public|private|protected)\s+(class|interface|enum)', 3),
                (r'\bimport\s+[\w\.]+\s*;', 2),
                (r'\bvoid\s+main\s*\(String\[\]', 3),
                (r'\bSystem\.out\.', 1),
                (r'\bthrows\s+', 1),
                (r'@Override', 1),
            ],
            'javascript': [
                (r'^\s*(function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+)', 2),
                (r'^\s*export\s+(default\s+)?', 2),
                (r'\bconsole\.(log|warn|error)\b', 1),
                (r'\bdocument\.', 1),
                (r'\bwindow\.', 1),
                (r'\brequire\(', 1),
                (r'`[^`]*\$\{[^}]*\}[^`]*`', 2),
            ],
            'php': [
                (r'^<\?php', 4),
                (r'\becho\s+', 2),
                (r'\$_[A-Z]+', 2),
                (r'\$\w+\s*=\s*', 1),
                (r'\bfunction\s+\w+\s*\(', 2),
                (r'->', 1),
            ],
            'ruby': [
                (r'^\s*def\s+\w+', 2),
                (r'^\s*class\s+\w+', 2),
                (r'^\s*module\s+\w+', 2),
                (r'\bputs\b', 1),
                (r'\brequire\b', 1),
                (r'\battr_(accessor|reader|writer)\b', 1),
                (r':\w+', 1),
            ],
            'go': [
                (r'^package\s+\w+', 3),
                (r'^import\s+"', 2),
                (r'\bfunc\s+(main|init)\s*\(', 3),
                (r':=', 2),
                (r'\berror\b', 1),
                (r'\bgoroutine\b', 1),
            ]
        }

        # Shebang patterns
        self.shebang_patterns = {
            r'^#!.*\bpython': 'python',
            r'^#!.*\bnode': 'javascript',
            r'^#!.*\bruby': 'ruby',
            r'^#!.*\bphp': 'php',
            r'^#!.*\bbash': 'shell',
            r'^#!.*\bsh': 'shell',
        }

    def detect_by_extension(self, filename: str) -> Optional[str]:
        """Detect language by file extension."""
        if not filename:
            return None

        filename_lower = filename.lower()

        # Check for exact extension matches
        for ext, lang in self.extension_to_language.items():
            if filename_lower.endswith(ext):
                return lang

        # Special case: C# files should return None
        if filename_lower.endswith('.cs'):
            return None

        return None

    def detect_by_shebang(self, code: str) -> Optional[str]:
        """Detect language from shebang line."""
        if not code:
            return None

        first_line = code.strip().split('\n')[0].strip()

        for pattern, language in self.shebang_patterns.items():
            if re.match(pattern, first_line, re.IGNORECASE):
                return language

        return None

    def detect_by_content(self, code: str) -> Tuple[Optional[str], float]:
        """Detect language by analyzing code content."""
        if not code or len(code.strip()) < 10:
            return None, 0.0

        # Check first for shebang
        shebang_lang = self.detect_by_shebang(code)
        if shebang_lang:
            return shebang_lang, 0.95

        # Analyze content patterns
        scores = {lang: 0.0 for lang in self.content_patterns.keys()}

        # Take first 100 lines for analysis
        lines = code.split('\n')[:100]
        sample_text = '\n'.join(lines)

        # Score each language based on patterns
        for language, patterns in self.content_patterns.items():
            for pattern, weight in patterns:
                matches = re.findall(pattern, sample_text, re.MULTILINE | re.IGNORECASE)
                scores[language] += len(matches) * weight

        # Get best match
        if not scores:
            return None, 0.0

        best_language = max(scores.items(), key=lambda x: x[1])

        # Calculate confidence
        total_score = sum(scores.values())
        if total_score == 0:
            return None, 0.0

        confidence = best_language[1] / max(10, total_score / 3)
        confidence = min(1.0, confidence)

        if best_language[1] >= 2:
            return best_language[0], confidence

        return None, 0.0  # Unsupported language

    def detect(self, filename: str = "", code: str = "") -> Tuple[Optional[str], float]:
        """
        Main detection method.
        IMPORTANT: Always returns None for .cs files
        """
        # FIX: If filename ends with .cs, return None immediately
        if filename and filename.lower().endswith('.cs'):
            return None, 0.0  # Unsupported language

        # Step 1: Detect by extension
        lang_by_ext = self.detect_by_extension(filename) if filename else None

        # Step 2: Detect by content
        lang_by_content, content_confidence = self.detect_by_content(code) if code else (None, 0.0)

        # Step 3: Combine results
        if lang_by_ext and lang_by_content:
            if lang_by_ext == lang_by_content:
                return lang_by_ext, max(0.9, content_confidence)
            else:
                if filename and '.' in filename:
                    return lang_by_ext, 0.7
                else:
                    return lang_by_content, content_confidence
        elif lang_by_ext:
            return lang_by_ext, 0.8
        elif lang_by_content:
            return lang_by_content, content_confidence
        else:
            return None, 0.0  # Unsupported language

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.content_patterns.keys())

    def get_language_for_extension(self, extension: str) -> Optional[str]:
        """Get language for a specific extension."""
        return self.extension_to_language.get(extension.lower())

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language.lower() in self.content_patterns


# Singleton instance
_detector_instance = None

def get_language_detector() -> LanguageDetector:
    """Get singleton instance of LanguageDetector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LanguageDetector()
    return _detector_instance

if __name__ == "__main__":
    # Simple test
    detector = LanguageDetector()
    print("🧪 Language Detector Tests")
    print("=" * 50)

    # Test basic detection
    test_cases = [
        ("test.py", "print('Hello')"),
        ("script.js", "console.log('Hello');"),
        ("Program.java", "public class Program { }"),
        ("test.cs", "using System; namespace Test { }"),
        ("unknown.txt", "Some random text"),
    ]

    for filename, code in test_cases:
        lang, conf = detector.detect(filename, code)
        print(f"{filename:20} -> {str(lang):15} (confidence: {conf:.2f})")

    print("\n✅ Supported languages:", ", ".join(detector.get_supported_languages()))
