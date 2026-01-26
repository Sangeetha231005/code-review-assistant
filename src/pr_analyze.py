#!/usr/bin/env python3
"""
PR Analysis Script - Analyzes ONLY PR-changed files
Part of the required PR flow
"""

import os
import sys
import json
import subprocess
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

def get_changed_files() -> List[str]:
    """
    Get list of files changed in the PR compared to main/master
    Uses: git diff --name-only origin/main...HEAD
    """
    try:
        # Try to get base branch from GitHub Actions env
        base_branch = os.environ.get("GITHUB_BASE_REF", "main")
        
        # Use git diff to get changed files
        cmd = f"git diff --name-only origin/{base_branch}...HEAD"
        print(f"🔍 Getting changed files with: {cmd}")
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode != 0:
            print(f"⚠️ Git diff failed: {result.stderr}")
            return []
        
        files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        
        # Filter only existing files and code files
        valid_extensions = ['.py', '.js', '.java', '.php', '.rb', '.go', '.c', '.cpp', '.cs']
        code_files = []
        
        for file in files:
            if not os.path.exists(file):
                print(f"⚠️ File doesn't exist: {file}")
                continue
                
            # Check if it's a code file
            if any(file.endswith(ext) for ext in valid_extensions):
                code_files.append(file)
            else:
                print(f"⚠️ Skipping non-code file: {file}")
        
        print(f"📋 Found {len(code_files)} changed code files:")
        for file in code_files:
            print(f"   - {file}")
            
        return code_files
        
    except Exception as e:
        print(f"❌ Error getting changed files: {e}")
        return []

def analyze_file(file_path: str) -> Dict[str, Any]:
    """
    Analyze a single file using the complete pipeline
    """
    try:
        # Import pipeline (lazy import to handle errors)
        from integration import CodeReviewPipeline
        
        print(f"\n{'='*60}")
        print(f"🚀 ANALYZING PR FILE: {file_path}")
        print(f"{'='*60}")
        
        pipeline = CodeReviewPipeline()
        result = pipeline.process_file(file_path)
        
        # Convert to dict for JSON output
        result_dict = result.to_dict()
        
        # Add file path
        result_dict['file_path'] = file_path
        
        return result_dict
        
    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'file_path': file_path,
            'error': str(e),
            'analysis_failed': True,
            'final_decision': 'ERROR'
        }

def main():
    """
    Main function for PR analysis
    """
    print("=" * 80)
    print("🔍 PR CODE ANALYSIS - Reviewing Changed Files Only")
    print("=" * 80)
    
    # Get changed files
    changed_files = get_changed_files()
    
    if not changed_files:
        print("ℹ️ No code files changed in this PR")
        
        # Create empty result
        summary = {
            'pr_analysis_summary': {
                'total_files': 0,
                'files_analyzed': 0,
                'overall_decision': 'NO_CHANGES',
                'decisions': {},
                'recommendations': ['No code files to analyze']
            }
        }
        
        print(json.dumps(summary, indent=2))
        return
    
    # Analyze each file
    results = []
    decisions_summary = []
    
    for file_path in changed_files:
        result = analyze_file(file_path)
        results.append(result)
        
        # Extract decision for summary
        decision = result.get('final_decision', 'UNKNOWN')
        filename = os.path.basename(file_path)
        
        decisions_summary.append({
            'file': filename,
            'decision': decision,
            'score': result.get('overall_score', 0),
            'language': result.get('language', 'unknown')
        })
        
        # Print quick summary
        print(f"📊 {filename}: {decision} (Score: {result.get('overall_score', 0)}/100)")
    
    # Calculate overall decision
    overall_decision = "APPROVE"
    
    # Check if any file should be rejected
    for result in results:
        if result.get('final_decision') == 'REJECT':
            overall_decision = 'REJECT'
            break
        elif result.get('final_decision') == 'REVIEW_REQUIRED':
            overall_decision = 'REVIEW_REQUIRED'
        elif result.get('final_decision') == 'REVIEW_RECOMMENDED' and overall_decision == 'APPROVE':
            overall_decision = 'REVIEW_RECOMMENDED'
    
    # Gather recommendations
    all_recommendations = []
    for result in results:
        recs = result.get('recommendations', [])
        filename = os.path.basename(result.get('file_path', 'unknown'))
        for rec in recs:
            all_recommendations.append(f"{filename}: {rec}")
    
    # Create final summary
    summary = {
        'pr_analysis_summary': {
            'total_files': len(changed_files),
            'files_analyzed': len([r for r in results if not r.get('analysis_failed', False)]),
            'overall_decision': overall_decision,
            'decisions': decisions_summary,
            'recommendations': all_recommendations,
            'detailed_results': results
        }
    }
    
    print(f"\n{'='*80}")
    print("📋 PR ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"📁 Total files: {len(changed_files)}")
    print(f"🔍 Analyzed: {summary['pr_analysis_summary']['files_analyzed']}")
    print(f"⚖️ Overall Decision: {overall_decision}")
    print(f"{'='*80}")
    
    # Print detailed decisions
    print("\n📊 File-by-File Decisions:")
    for decision in decisions_summary:
        symbol = ""
        if decision['decision'] == 'APPROVE':
            symbol = "✅"
        elif decision['decision'] == 'REVIEW_RECOMMENDED':
            symbol = "⚠️"
        elif decision['decision'] == 'REVIEW_REQUIRED':
            symbol = "🔴"
        elif decision['decision'] == 'REJECT':
            symbol = "❌"
        
        print(f"  {symbol} {decision['file']}: {decision['decision']} (Score: {decision['score']}/100)")
    
    # Print recommendations
    if all_recommendations:
        print(f"\n💡 Recommendations ({len(all_recommendations)}):")
        for i, rec in enumerate(all_recommendations[:5], 1):  # Limit to 5
            print(f"  {i}. {rec}")
        if len(all_recommendations) > 5:
            print(f"  ... and {len(all_recommendations) - 5} more")
    
    print(f"\n{'='*80}")
    print("FINAL PR DECISION:")
    
    if overall_decision == "APPROVE":
        print("✅ APPROVE - Code is safe to merge")
    elif overall_decision == "REVIEW_RECOMMENDED":
        print("⚠️ REVIEW RECOMMENDED - Consider reviewing before merging")
    elif overall_decision == "REVIEW_REQUIRED":
        print("🔴 REVIEW REQUIRED - Security review needed before merging")
    elif overall_decision == "REJECT":
        print("❌ REJECT - Code has security issues")
    else:
        print("❓ UNKNOWN - Could not determine decision")
    
    print(f"{'='*80}")
    
    # Output JSON for GitHub Actions
    print("\n📤 JSON Output for CI:")
    print(json.dumps(summary, indent=2))
    
    # Exit with appropriate code for CI
    if overall_decision == "REJECT":
        print("\n❌ PR will be marked as failed due to security issues")
        sys.exit(1)
    elif overall_decision == "REVIEW_REQUIRED":
        print("\n⚠️ PR requires manual review")
        sys.exit(0)  # Don't fail, but require review
    else:
        print("\n✅ PR analysis completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()
