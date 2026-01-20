"""
Test script for the complete integration pipeline
"""
import os
import sys
import tempfile

# Add project to path
project_root = "/content/drive/MyDrive/code-review-assistant"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from integration import CodeReviewPipeline

def create_test_file(content, suffix='.py'):
    """Helper to create temporary test files"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name

def analyze_code_with_pipeline(code_content, file_suffix='.py'):
    """Helper to analyze code with the pipeline"""
    temp_file = None
    try:
        temp_file = create_test_file(code_content, file_suffix)
        pipeline = CodeReviewPipeline()
        result = pipeline.process_file(temp_file)
        return result
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

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

    print("Code contains: Command injection + SQL injection vulnerabilities")
    
    # Analyze code
    result = analyze_code_with_pipeline(vulnerable_code, '.py')
    
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

    print("Code is safe with input sanitization")
    
    # Analyze code
    result = analyze_code_with_pipeline(safe_code, '.py')
    
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

    print("JavaScript code with XSS vulnerability")
    
    # Analyze code
    result = analyze_code_with_pipeline(js_code, '.js')
    
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

    print("C++ code (unsupported language)")
    
    # Analyze code
    result = analyze_code_with_pipeline(cpp_code, '.cpp')
    
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

def test_actual_vulnerability_model_missing():
    """Test actual vulnerability model missing scenario - only fails if pipeline detects it"""
    print("\n" + "=" * 80)
    print("🧪 TEST 5: VULNERABILITY MODEL INTEGRATION")
    print("=" * 80)
    
    # This test only fails if the pipeline actually reports the model is missing
    # We'll test with a simple Python file to see if the pipeline works
    
    test_code = '''print("Hello World")'''
    
    try:
        result = analyze_code_with_pipeline(test_code, '.py')
        
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS:")
        print("=" * 80)
        print(f"Language: {result.language}")
        print(f"Security Analysis: {result.security_analysis.status}")
        print(f"Final Decision: {result.final_decision}")
        
        # If the pipeline reports model missing, we should see it in the status
        if result.security_analysis.status == "MODEL_MISSING":
            print("\n⚠️  Pipeline reported vulnerability model is missing")
            print("This is a valid test failure - the pipeline correctly detected the issue")
            return {"status": "MODEL_MISSING", "decision": "FAIL"}
        else:
            print("\n✅ Vulnerability model integration working")
            return result
            
    except Exception as e:
        # If the pipeline itself crashes due to missing model, that's a different kind of failure
        error_msg = str(e).lower()
        if "model" in error_msg or "vulnerability" in error_msg:
            print(f"\n⚠️  Pipeline error (likely missing model): {e}")
            return {"status": "PIPELINE_ERROR", "decision": "FAIL"}
        else:
            raise

def run_all_tests():
    """Run all integration tests and collect results"""
    print("=" * 80)
    print("🚀 INTEGRATION TEST SUITE - CODE REVIEW PIPELINE")
    print("=" * 80)

    test_results = []
    failed_tests = []

    try:
        # Test 1: Vulnerable Python
        result1 = test_vulnerable_python()
        test_results.append(("Vulnerable Python", "PASS", result1.final_decision))
        if result1.final_decision in ["REJECT", "REVIEW_REQUIRED"]:
            failed_tests.append(("Vulnerable Python", result1.final_decision))
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        test_results.append(("Vulnerable Python", "FAIL", str(e)))
        failed_tests.append(("Vulnerable Python", f"ERROR: {e}"))

    try:
        # Test 2: Safe Python
        result2 = test_safe_python()
        test_results.append(("Safe Python", "PASS", result2.final_decision))
        if result2.final_decision in ["REJECT", "REVIEW_REQUIRED"]:
            failed_tests.append(("Safe Python", result2.final_decision))
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        test_results.append(("Safe Python", "FAIL", str(e)))

    try:
        # Test 3: JavaScript
        result3 = test_javascript()
        test_results.append(("JavaScript", "PASS", result3.final_decision))
        if result3.final_decision in ["REJECT", "REVIEW_REQUIRED"]:
            failed_tests.append(("JavaScript", result3.final_decision))
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        test_results.append(("JavaScript", "FAIL", str(e)))

    try:
        # Test 4: Unsupported Language
        result4 = test_unsupported_language()
        test_results.append(("Unsupported Language", "PASS", result4.final_decision))
        if result4.final_decision in ["REJECT", "REVIEW_REQUIRED"]:
            failed_tests.append(("Unsupported Language", result4.final_decision))
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        test_results.append(("Unsupported Language", "FAIL", str(e)))

    try:
        # Test 5: Vulnerability Model Integration
        result5 = test_actual_vulnerability_model_missing()
        if isinstance(result5, dict) and result5.get("status") in ["MODEL_MISSING", "PIPELINE_ERROR"]:
            test_results.append(("Vulnerability Model", "FAIL", result5["status"]))
            failed_tests.append(("Vulnerability Model", "MODEL_MISSING"))
        else:
            test_results.append(("Vulnerability Model", "PASS", "Integration OK"))
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
        test_results.append(("Vulnerability Model", "FAIL", str(e)))

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

    # CI Enforcement Logic - Happens AFTER all tests complete
    print("\n" + "=" * 80)
    print("🔒 CI ENFORCEMENT CHECK")
    print("=" * 80)
    
    if failed_tests:
        print("\n❌ SECURITY ISSUES DETECTED:")
        for test_name, decision in failed_tests:
            print(f"  • {test_name}: {decision}")
        
        print("\n🚫 CI PIPELINE BLOCKED - Vulnerabilities detected")
        print("This would trigger: sys.exit(1) in CI/CD")
        
        # In a real CI pipeline, we would exit with code 1 here
        # For demonstration, we'll just print the action
        print("\n[CI ACTION] sys.exit(1) - Pipeline failed due to security issues")
    else:
        print("\n✅ All tests passed - No security issues detected")
        print("This would trigger: sys.exit(0) in CI/CD")
        print("\n[CI ACTION] sys.exit(0) - Pipeline passed")

    return test_results, failed_tests

def ci_main():
    """Main function for CI pipeline - enforces security checks"""
    print("=" * 80)
    print("🔒 CI/CD SECURITY ENFORCEMENT PIPELINE")
    print("=" * 80)
    
    # Run tests and collect security decisions
    _, failed_tests = run_all_tests()
    
    # Real CI enforcement - exit with appropriate code
    if failed_tests:
        print("\n" + "=" * 80)
        print("❌ CI PIPELINE FAILED - Blocking merge")
        print("=" * 80)
        sys.exit(1)  # This is the actual CI enforcement
    else:
        print("\n" + "=" * 80)
        print("✅ CI PIPELINE PASSED - Allowing merge")
        print("=" * 80)
        sys.exit(0)

if __name__ == "__main__":
    # For normal test runs, just show results
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        ci_main()
    else:
        print("Running in test mode (use --ci flag for CI enforcement)")
        print("-" * 80)
        run_all_tests()
