"""
Main Style Checker Module
Orchestrates language detection and linting for 6 languages only
"""
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class AnalysisStatus(Enum):
    """Status of code analysis."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    filename: str
    language: str
    confidence: float
    status: AnalysisStatus
    passed: bool
    score: int
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    info: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    raw_output: str
    processing_time: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['status'] = self.status.value
        return result


class StyleChecker:
    """Main style checker class for 6 languages."""
    def __init__(self):
        from language_detector import get_language_detector
        from linter_runner import get_linter_runner

        self.detector = get_language_detector()
        self.linter = get_linter_runner()

        # Statistics
        self.stats = {
            'total_files': 0,
            'passed_files': 0,
            'failed_files': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'total_processing_time': 0.0
        }

    def analyze_code(self, code: str, filename: str = "unknown.txt") -> AnalysisResult:
        """
        Analyze code for style issues.

        Args:
            code: Source code to analyze
            filename: Original filename (for language detection)

        Returns:
            AnalysisResult object
        """
        start_time = time.time()

        # Validate input
        if not code or not code.strip():
            return AnalysisResult(
                filename=filename,
                language="unknown",
                confidence=0.0,
                status=AnalysisStatus.ERROR,
                passed=False,
                score=0,
                errors=[{
                    'line': 1,
                    'message': 'No code provided',
                    'severity': 'error',
                    'tool': 'system'
                }],
                warnings=[],
                info=[],
                metrics={},
                raw_output='',
                processing_time=0.0,
                timestamp=start_time
            )

        if len(code.strip()) < 5:
            return AnalysisResult(
                filename=filename,
                language="unknown",
                confidence=0.0,
                status=AnalysisStatus.ERROR,
                passed=False,
                score=0,
                errors=[{
                    'line': 1,
                    'message': 'Code too short for analysis (minimum 5 characters)',
                    'severity': 'error',
                    'tool': 'system'
                }],
                warnings=[],
                info=[],
                metrics={},
                raw_output='',
                processing_time=0.0,
                timestamp=start_time
            )

        try:
            # Step 1: Detect language
            language, confidence = self.detector.detect(filename, code)

            if not language:
                return AnalysisResult(
                    filename=filename,
                    language="unknown",
                    confidence=0.0,
                    status=AnalysisStatus.ERROR,
                    passed=False,
                    score=0,
                    errors=[{
                        'line': 1,
                        'message': 'Could not detect programming language',
                        'severity': 'error',
                        'tool': 'detector'
                    }],
                    warnings=[],
                    info=[],
                    metrics={},
                    raw_output='',
                    processing_time=0.0,
                    timestamp=start_time
                )

            # Step 2: Check if language is supported
            if not self.detector.is_language_supported(language):
                return AnalysisResult(
                    filename=filename,
                    language=language,
                    confidence=confidence,
                    status=AnalysisStatus.WARNING,
                    passed=True,  # Pass by default for unsupported languages
                    score=80,  # Default score for unsupported
                    errors=[],
                    warnings=[{
                        'line': 1,
                        'message': f'Language "{language}" is not in supported list (Python, Java, JavaScript, PHP, Ruby, Go)',
                        'severity': 'warning',
                        'tool': 'system'
                    }],
                    info=[],
                    metrics={'lines_of_code': len(code.split('\n'))},
                    raw_output='',
                    processing_time=0.0,
                    timestamp=start_time
                )

            # Step 3: Run linter
            linter_result = self.linter.run_linter(language, code, filename)

            # Step 4: Process results
            processing_time = time.time() - start_time

            result = AnalysisResult(
                filename=filename,
                language=language,
                confidence=confidence,
                status=AnalysisStatus.SUCCESS,
                passed=linter_result['passed'],
                score=linter_result['score'],
                errors=linter_result['errors'],
                warnings=linter_result['warnings'],
                info=linter_result.get('info', []),
                metrics=linter_result.get('metrics', {}),
                raw_output=linter_result.get('raw_output', ''),
                processing_time=processing_time,
                timestamp=start_time
            )

            # Update statistics
            self._update_stats(result)

            return result

        except Exception as e:
            processing_time = time.time() - start_time

            return AnalysisResult(
                filename=filename,
                language="unknown",
                confidence=0.0,
                status=AnalysisStatus.ERROR,
                passed=False,
                score=0,
                errors=[{
                    'line': 1,
                    'message': f'Analysis failed: {str(e)}',
                    'severity': 'error',
                    'tool': 'system'
                }],
                warnings=[],
                info=[],
                metrics={},
                raw_output=str(e),
                processing_time=processing_time,
                timestamp=start_time
            )

    def _update_stats(self, result: AnalysisResult):
        """Update statistics with new result."""
        self.stats['total_files'] += 1
        self.stats['total_processing_time'] += result.processing_time

        if result.passed:
            self.stats['passed_files'] += 1
        else:
            self.stats['failed_files'] += 1

        self.stats['total_errors'] += len(result.errors)
        self.stats['total_warnings'] += len(result.warnings)

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = self.stats.copy()
        if stats['total_files'] > 0:
            stats['average_processing_time'] = stats['total_processing_time'] / stats['total_files']
            stats['pass_rate'] = (stats['passed_files'] / stats['total_files']) * 100
        else:
            stats['average_processing_time'] = 0.0
            stats['pass_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """Reset all statistics."""
        self.stats = {
            'total_files': 0,
            'passed_files': 0,
            'failed_files': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'total_processing_time': 0.0
        }

    def analyze_multiple_files(self, files: List[Dict[str, str]]) -> List[AnalysisResult]:
        """
        Analyze multiple files.

        Args:
            files: List of dicts with 'filename' and 'code' keys

        Returns:
            List of AnalysisResult objects
        """
        results = []

        for file_info in files:
            filename = file_info.get('filename', 'unknown.txt')
            code = file_info.get('code', '')

            result = self.analyze_code(code, filename)
            results.append(result)

        return results

    def get_summary_report(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        Generate a summary report for multiple analysis results.

        Args:
            results: List of AnalysisResult objects

        Returns:
            Summary dictionary
        """
        if not results:
            return {
                'total_files': 0,
                'passed_files': 0,
                'failed_files': 0,
                'total_errors': 0,
                'total_warnings': 0,
                'total_info': 0,
                'average_score': 0,
                'overall_status': 'unknown'
            }

        total_files = len(results)
        passed_files = sum(1 for r in results if r.passed)
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        total_info = sum(len(r.info) for r in results)
        average_score = sum(r.score for r in results) / total_files

        # Determine overall status
        if total_errors == 0 and total_warnings == 0:
            overall_status = 'excellent'
        elif total_errors == 0:
            overall_status = 'good'
        elif passed_files == total_files:
            overall_status = 'acceptable'
        else:
            overall_status = 'needs_improvement'

        # Language distribution
        language_dist = {}
        for result in results:
            lang = result.language
            language_dist[lang] = language_dist.get(lang, 0) + 1

        return {
            'total_files': total_files,
            'passed_files': passed_files,
            'failed_files': total_files - passed_files,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'total_info': total_info,
            'average_score': round(average_score, 2),
            'overall_status': overall_status,
            'language_distribution': language_dist,
            'pass_rate': round((passed_files / total_files) * 100, 2) if total_files > 0 else 0
        }

    def format_report(self, result: AnalysisResult, detailed: bool = False) -> str:
        """
        Format analysis result as a readable report.

        Args:
            result: AnalysisResult object
            detailed: Whether to include detailed information

        Returns:
            Formatted report string
        """
        report = []

        # Header
        report.append("=" * 70)
        report.append("📋 CODE STYLE ANALYSIS REPORT")
        report.append("=" * 70)

        # File information
        report.append(f"\n📁 File: {result.filename}")
        report.append(f"🔤 Language: {result.language}")
        report.append(f"📈 Detection Confidence: {result.confidence:.1%}")
        report.append(f"⏱️  Processing Time: {result.processing_time:.3f} seconds")

        # Summary
        report.append(f"\n📊 SUMMARY")
        report.append(f"  Status: {'✅ PASSED' if result.passed else '❌ NEEDS IMPROVEMENT'}")
        report.append(f"  Score: {result.score}/100")
        report.append(f"  Errors: {len(result.errors)}")
        report.append(f"  Warnings: {len(result.warnings)}")
        report.append(f"  Info: {len(result.info)}")

        # Metrics
        if result.metrics:
            report.append(f"\n📈 METRICS")
            for key, value in result.metrics.items():
                report.append(f"  {key.replace('_', ' ').title()}: {value}")

        # Detailed errors
        if detailed and result.errors:
            report.append(f"\n❌ ERRORS ({len(result.errors)}):")
            for i, error in enumerate(result.errors[:20], 1):
                tool = error.get('tool', 'unknown')
                line = error.get('line', '?')
                message = error.get('message', 'Unknown error')
                report.append(f"  {i:2d}. [{tool}] Line {line}: {message}")

        # Detailed warnings
        if detailed and result.warnings:
            report.append(f"\n⚠️  WARNINGS ({len(result.warnings)}):")
            for i, warning in enumerate(result.warnings[:20], 1):
                tool = warning.get('tool', 'unknown')
                line = warning.get('line', '?')
                message = warning.get('message', 'Unknown warning')
                report.append(f"  {i:2d}. [{tool}] Line {line}: {message}")

        # Footer
        report.append("\n" + "=" * 70)

        return "\n".join(report)

    def export_results(self, result: AnalysisResult, format: str = 'json') -> str:
        """
        Export analysis results in different formats.

        Args:
            result: AnalysisResult object
            format: Output format ('json', 'text')

        Returns:
            Formatted output string
        """
        if format == 'json':
            return json.dumps(result.to_dict(), indent=2)
        elif format == 'text':
            return self.format_report(result, detailed=True)
        else:
            return f"Unsupported format: {format}"


# Singleton instance
_checker_instance = None

def get_style_checker() -> StyleChecker:
    """Get singleton instance of StyleChecker."""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = StyleChecker()
    return _checker_instance

if __name__ == "__main__":
    # Test the style checker
    checker = StyleChecker()

    test_code = """
def calculate_sum(a, b):
    '''Calculate the sum of two numbers.'''
    return a + b

def main():
    x = 5
    y = 10
    result = calculate_sum(x, y)
    print(f"The sum is: {result}")

if __name__ == "__main__":
    main()
"""
    result = checker.analyze_code(test_code, "test.py")
    print(checker.format_report(result, detailed=True))
