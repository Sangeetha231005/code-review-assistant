"""
Test script for the complete integration pipeline
(CI-safe version: vulnerable tests do NOT block PRs)
"""

import os
import sys
import tempfile

# Add project to path
project_root = "/content/drive/MyDrive/code-review-assistant"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from integration import CodeReviewPipeline


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def create_test_file(content, suffix):
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name


def analyze_code(code, suffix):
    temp_file = None
    try:
        temp_file = create_test_file(code, suffix)
        pipeline = CodeReviewPipeline()
        return pipeline.process_file(temp_file)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


# -------------------------------------------------------------------
# Test 1: Vulnerable Python (EXPECTED REJECT → PASS)
# -------------------------------------------------------------------

def test_vulnerable_python():
    print("=" * 80)
    print("🧪 TEST 1: VULNERABLE PYTHON (EXPECTED REJECT)")
    print("=" * 80)

    code = '''
import os
def bad(user_input):
    os.system("echo " + user_input)
'''

    result = analyze_code(code, ".py")

    print("Decision:", result.final_decision)

    assert result.language.lower() == "python"
    assert result.final_decision in ["REJECT", "REVIEW_REQUIRED"]

    print("✅ Vulnerable Python correctly rejected")
    return result


# -------------------------------------------------------------------
# Test 2: Safe Python (MUST APPROVE)
# -------------------------------------------------------------------

def test_safe_python():
    print("=" * 80)
    print("🧪 TEST 2: SAFE PYTHON")
    print("=" * 80)

    code = '''
import re
def sanitize(x):
    return re.sub(r"[^a-zA-Z0-9]", "", x)
print(sanitize("test123"))
'''

    result = analyze_code(code, ".py")

    print("Decision:", result.final_decision)

    assert result.language.lower() == "python"
    assert result.final_decision in ["APPROVE", "REVIEW_RECOMMENDED"]

    print("✅ Safe Python approved")
    return result


# -------------------------------------------------------------------
# Test 3: Vulnerable JavaScript (EXPECTED REJECT → PASS)
# -------------------------------------------------------------------

def test_javascript():
    print("=" * 80)
    print("🧪 TEST 3: JAVASCRIPT XSS (EXPECTED REJECT)")
    print("=" * 80)

    code = '''
document.write("<div>" + userInput + "</div>");
'''

    result = analyze_code(code, ".js")

    print("Decision:", result.final_decision)

    assert result.language.lower() == "javascript"
    assert result.final_decision in ["REJECT", "REVIEW_REQUIRED"]

    print("✅ JavaScript vulnerability detected")
    return result


# -------------------------------------------------------------------
# Test 4: Unsupported Language (EXPECTED REJECT → PASS)
# -------------------------------------------------------------------

def test_unsupported_language():
    print("=" * 80)
    print("🧪 TEST 4: UNSUPPORTED LANGUAGE")
    print("=" * 80)

    code = "int main(){ return 0; }"

    result = analyze_code(code, ".cpp")

    print("Decision:", result.final_decision)

    assert result.overall_status == "UNSUPPORTED_LANGUAGE"
    assert result.final_decision == "REJECT"

    print("✅ Unsupported language rejected")
    return result


# -------------------------------------------------------------------
# Test 5: Model Missing (WARNING ONLY)
# -------------------------------------------------------------------

def test_model_presence():
    print("=" * 80)
    print("🧪 TEST 5: MODEL CHECK (NON-BLOCKING)")
    print("=" * 80)

    result = analyze_code('print("hello")', ".py")

    status = result.security_analysis.status
    print("Model status:", status)

    if status == "MODEL_MISSING":
        print("⚠️ Model missing — warning only (CI not blocked)")
    else:
        print("✅ Model loaded")

    return result


# -------------------------------------------------------------------
# Test Runner
# -------------------------------------------------------------------

def run_all_tests():
    print("=" * 80)
    print("🚀 INTEGRATION TEST SUITE (CI-SAFE)")
    print("=" * 80)

    failed = []

    try:
        test_vulnerable_python()
        test_safe_python()
        test_javascript()
        test_unsupported_language()
        test_model_presence()
    except AssertionError as e:
        failed.append(str(e))

    print("=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)

    if failed:
        print("❌ CI FAILED — pipeline behaved incorrectly")
        for f in failed:
            print(" •", f)
        sys.exit(1)
    else:
        print("✅ CI PASSED — all expectations met")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
