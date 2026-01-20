"""
Test script for the complete integration pipeline
"""
print("🚀 LIVE DEMO: AUTOMATED CODE REVIEW PIPELINE")
import os
import sys
import tempfile

# Add project to path
project_root = "/content/drive/MyDrive/code-review-assistant"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from integration import CodeReviewPipeline

def test_vulnerable_python():
    """Test with vulnerable Python code"""
    print("=" * 80)
    print("🧪 TEST 1: VULNERABLE PYTHON CODE")
    print("=" * 80)

    vulnerable_code = '''import os
import subprocess

def vulnerable_function(user_input):
    # Command injection vulnerability
    os.system("echo " + user_input)
    return "Done"

def execute_query(query):
    # SQL injection vulnerability
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("SELECT * FROM users WHERE id = " + query)  # SQL injection
    return "Query executed"

def main():
    user_data = input("Enter your name: ")
    vulnerable_function(user_data)

    user_query = input("Enter user ID: ")
    execute_query(user_query)

if __name__ == "__main__":
    main()
'''

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(vulnerable_code)
        temp_file = f.name

    print(f"Created test file: {temp_file}")
    print("Code contains: Command injection + SQL injection vulnerabilities")

    # Run pipeline
    pipeline = CodeReviewPipeline()
    result = pipeline.process_file(temp_file)

    # Clean up
    os.unlink(temp_file)

    print("\n" + "=" * 80)
    print("📊 TEST RESULTS:")
    print("=" * 80)
    print(f"Language: {result.language}")
    print(f"Style Score: {result.style_analysis.data.get('score', 0)}/100")
    print(f"Security Score: {result.security_analysis.data.get('score', 0)}/100")
    print(f"Overall Score: {result.overall_score}/100")
    print(f"Final Decision: {result.final_decision}")
    print(f"Status: {result.overall_status}")

    # Verify expectations
    assert result.language.lower() == "python", f"Expected python, got {result.language}"
    assert result.final_decision in ["REJECT", "REVIEW_REQUIRED"], \
        f"Expected REJECT or REVIEW_REQUIRED for vulnerable code, got {result.final_decision}"

    print("\n✅ TEST PASSED: Vulnerable code correctly identified!")
    return result

def test_safe_python():
    """Test with safe Python code"""
    print("\n" + "=" * 80)
    print("🧪 TEST 2: SAFE PYTHON CODE")
    print("=" * 80)

    safe_code = '''import json
import re

def sanitize_input(user_input):
    """Sanitize user input"""
    # Remove any non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9]', '', user_input)

def safe_function(user_input):
    """Safe function with input validation"""
    sanitized = sanitize_input(user_input)
    print(f"Sanitized output: {sanitized}")
    return sanitized

def calculate_sum(a, b):
    """Calculate sum of two numbers"""
    return a + b

def main():
    # Safe operations
    result = calculate_sum(5, 10)
    print(f"Sum: {result}")

    # Safe with sanitization
    user_data = "test123"
    safe_function(user_data)

    # Safe file operations
    with open("temp.txt", "w") as f:
        f.write("Safe content")

if __name__ == "__main__":
    main()
'''

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(safe_code)
        temp_file = f.name

    print(f"Created test file: {temp_file}")
    print("Code is safe with input sanitization")

    # Run pipeline
    pipeline = CodeReviewPipeline()
    result = pipeline.process_file(temp_file)

    # Clean up
    os.unlink(temp_file)

    print("\n" + "=" * 80)
    print("📊 TEST RESULTS:")
    print("=" * 80)
    print(f"Language: {result.language}")
    print(f"Style Score: {result.style_analysis.data.get('score', 0)}/100")
    print(f"Security Score: {result.security_analysis.data.get('score', 0)}/100")
    print(f"Overall Score: {result.overall_score}/100")
    print(f"Final Decision: {result.final_decision}")
    print(f"Status: {result.overall_status}")

    # Verify expectations
    assert result.language.lower() == "python", f"Expected python, got {result.language}"
    assert result.final_decision in ["APPROVE", "REVIEW_RECOMMENDED"], \
        f"Expected APPROVE or REVIEW_RECOMMENDED for safe code, got {result.final_decision}"

    print("\n✅ TEST PASSED: Safe code correctly identified!")
    return result

def test_javascript():
    """Test with JavaScript code"""
    print("\n" + "=" * 80)
    print("🧪 TEST 3: JAVASCRIPT CODE")
    print("=" * 80)

    js_code = '''// JavaScript test file
function calculateSum(a, b) {
    return a + b;
}

function vulnerableFunction(userInput) {
    // XSS vulnerability
    document.write("<div>" + userInput + "</div>");
}

function safeFunction(userInput) {
    // Safe with sanitization
    const sanitized = userInput.replace(/[<>]/g, '');
    document.getElementById("output").innerText = sanitized;
}

// Main execution
const result = calculateSum(5, 10);
console.log("Sum:", result);

const userData = prompt("Enter data:");
vulnerableFunction(userData);
safeFunction(userData);
'''

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(js_code)
        temp_file = f.name

    print(f"Created test file: {temp_file}")
    print("JavaScript code with XSS vulnerability")

    # Run pipeline
    pipeline = CodeReviewPipeline()
    result = pipeline.process_file(temp_file)

    # Clean up
    os.unlink(temp_file)

    print("\n" + "=" * 80)
    print("📊 TEST RESULTS:")
    print("=" * 80)
    print(f"Language: {result.language}")
    print(f"Style Score: {result.style_analysis.data.get('score', 0)}/100")
    print(f"Security Score: {result.security_analysis.data.get('score', 0)}/100")
    print(f"Overall Score: {result.overall_score}/100")
    print(f"Final Decision: {result.final_decision}")

    print("\n✅ TEST PASSED: JavaScript code analyzed!")
    return result

def test_unsupported_language():
    """Test with unsupported language"""
    print("\n" + "=" * 80)
    print("🧪 TEST 4: UNSUPPORTED LANGUAGE (C++)")
    print("=" * 80)

    cpp_code = '''#include <iostream>
#include <string>

int main() {
    std::string user_input;
    std::cout << "Enter input: ";
    std::cin >> user_input;

    // Potentially unsafe system call
    std::string command = "echo " + user_input;
    system(command.c_str());

    return 0;
}
'''

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(cpp_code)
        temp_file = f.name

    print(f"Created test file: {temp_file}")
    print("C++ code (unsupported language)")

    # Run pipeline
    pipeline = CodeReviewPipeline()
    result = pipeline.process_file(temp_file)

    # Clean up
    os.unlink(temp_file)

    print("\n" + "=" * 80)
    print("📊 TEST RESULTS:")
    print("=" * 80)
    print(f"Language: {result.language}")
    print(f"Overall Status: {result.overall_status}")
    print(f"Final Decision: {result.final_decision}")

    # Verify expectations
    assert result.overall_status == "UNSUPPORTED_LANGUAGE", \
        f"Expected UNSUPPORTED_LANGUAGE for C++, got {result.overall_status}"
    assert result.final_decision == "REJECT", \
        f"Expected REJECT for unsupported language, got {result.final_decision}"

    print("\n✅ TEST PASSED: Unsupported language correctly handled!")
    return result

def run_all_tests():
    """Run all integration tests"""
    print("=" * 80)
    print("🚀 INTEGRATION TEST SUITE - CODE REVIEW PIPELINE")
    print("=" * 80)

    test_results = []

    try:
        # Test 1: Vulnerable Python
        result1 = test_vulnerable_python()
        test_results.append(("Vulnerable Python", "PASS", result1.final_decision))
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        test_results.append(("Vulnerable Python", "FAIL", str(e)))

    try:
        # Test 2: Safe Python
        result2 = test_safe_python()
        test_results.append(("Safe Python", "PASS", result2.final_decision))
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        test_results.append(("Safe Python", "FAIL", str(e)))

    try:
        # Test 3: JavaScript
        result3 = test_javascript()
        test_results.append(("JavaScript", "PASS", result3.final_decision))
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        test_results.append(("JavaScript", "FAIL", str(e)))

    try:
        # Test 4: Unsupported Language
        result4 = test_unsupported_language()
        test_results.append(("Unsupported Language", "PASS", result4.final_decision))
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        test_results.append(("Unsupported Language", "FAIL", str(e)))

    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUITE SUMMARY")
    print("=" * 80)

    for i, (test_name, status, details) in enumerate(test_results, 1):
        emoji = "✅" if status == "PASS" else "❌"
        print(f"{emoji} Test {i}: {test_name} - {status} - {details}")

    passed = sum(1 for _, status, _ in test_results if status == "PASS")
    total = len(test_results)

    print(f"\n📊 Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Integration pipeline is working correctly!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Check the errors above.")

    return test_results

if __name__ == "__main__":
    run_all_tests()
