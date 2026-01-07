"""
Input File Processor for Code Review Assistant
Reads and processes input files of different types
"""

import os
import json
import sys
import tempfile
from typing import List, Dict, Any, Optional

class InputFileProcessor:
   """Process different types of input files for code review"""
   
   @staticmethod
   def read_single_code_file(filepath: str) -> Dict[str, str]:
       """
       Read a single source code file
       Returns: {'filename': filename, 'code': code_content}
       """
       with open(filepath, 'r', encoding='utf-8') as f:
           code = f.read()
       
       filename = os.path.basename(filepath)
       return {'filename': filename, 'code': code}
   
   @staticmethod
   def read_json_input(json_filepath: str) -> List[Dict[str, str]]:
       """
       Read input from JSON file with format:
       [
           {"filename": "example.py", "code": "print('hello')"},
           {"filename": "test.js", "code": "console.log('test');"}
       ]
       """
       with open(json_filepath, 'r', encoding='utf-8') as f:
           data = json.load(f)
       
       if not isinstance(data, list):
           raise ValueError("JSON input must be a list of file objects")
       
       return data
   
   @staticmethod
   def read_directory(directory_path: str, extensions: List[str] = None) -> List[Dict[str, str]]:
       """
       Read all code files from a directory
       """
       files = []
       
       for root, _, filenames in os.walk(directory_path):
           for filename in filenames:
               if extensions:
                   if not any(filename.endswith(ext) for ext in extensions):
                       continue
               
               filepath = os.path.join(root, filename)
               try:
                   with open(filepath, 'r', encoding='utf-8') as f:
                       code = f.read()
                   
                   # Get relative path for better readability
                   rel_path = os.path.relpath(filepath, directory_path)
                   files.append({'filename': rel_path, 'code': code})
               except Exception as e:
                   print(f"⚠️ Could not read {filepath}: {e}")
       
       return files
   
   @staticmethod
   def create_test_input() -> List[Dict[str, str]]:
       """
       Create test input with multiple language examples
       """
       test_files = [
           {
               'filename': 'test_python.py',
               'code': '''
import os
import subprocess

def calculate_sum(a, b):
   """Calculate sum of two numbers."""
   return a + b

def vulnerable_function(user_input):
   # Vulnerable: command injection
   os.system(f"echo {user_input}")
   return "Done"

def main():
   x = 5
   y = 10
   result = calculate_sum(x, y)
   print(f"Sum: {result}")

if __name__ == "__main__":
   main()
'''
           },
           {
               'filename': 'test_javascript.js',
               'code': '''
function calculateSum(a, b) {
   return a + b;
}

// Potentially vulnerable XSS
function unsafeOutput(input) {
   document.getElementById('output').innerHTML = input;
}

// Safe output
function safeOutput(input) {
   document.getElementById('output').textContent = input;
}

// Main
const result = calculateSum(5, 10);
console.log(`Sum: ${result}`);
'''
           },
           {
               'filename': 'test_java.java',
               'code': '''
import java.sql.*;

public class Example {
   // Vulnerable SQL injection
   public void vulnerableQuery(String userId) {
       String query = "SELECT * FROM users WHERE id = " + userId;
       Statement stmt = connection.createStatement();
       ResultSet rs = stmt.executeQuery(query);
   }
   
   // Safe prepared statement
   public void safeQuery(String userId) {
       String query = "SELECT * FROM users WHERE id = ?";
       PreparedStatement pstmt = connection.prepareStatement(query);
       pstmt.setString(1, userId);
       ResultSet rs = pstmt.executeQuery();
   }
}
'''
           },
           {
               'filename': 'test_unsupported.cs',
               'code': '''
using System;

namespace TestApp
{
   class Program
   {
       static void Main(string[] args)
       {
           Console.WriteLine("This is C# - unsupported language");
       }
   }
}
'''
           }
       ]
       return test_files

if __name__ == "__main__":
   # Test the processor
   processor = InputFileProcessor()
   
   # Create test files
   test_files = processor.create_test_input()
   
   print(f"Created {len(test_files)} test files:")
   for file_info in test_files:
       print(f"  - {file_info['filename']} ({len(file_info['code'])} chars)")
