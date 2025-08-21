"""
Convenience script to run all demonstrations in sequence.
This script executes all five demo scripts and provides a summary of results.
"""
import subprocess
import sys
import os
import time
from pathlib import Path

from demo_utils import get_base_dir

# Define base directory as the parent directory of where this script is located
BASE_DIR = get_base_dir()

def find_file(filename):
    """Find a file in the same directory as this script or Code/ subdirectory"""
    # Check same directory as this script
    same_dir_path = os.path.join(BASE_DIR, "src", filename)
    if os.path.exists(same_dir_path):
        return same_dir_path
    
    code_path = os.path.join(BASE_DIR, "docs", filename)
    print(f"Looking for {filename} in {code_path}")
    if os.path.exists(code_path):
        return code_path
    
    return None

def setup_results_directory(demo_name):
    """Create results directory structure for a demo"""
    results_dir = Path(BASE_DIR) / "results" / demo_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def run_demo(demo_name):
    """Run a single demo script and capture its output"""
    print(f"\n{'='*60}")
    print(f"Running {demo_name}")
    print(f"{'='*60}")
    
    # Setup results directory
    results_dir = setup_results_directory(demo_name)
    print(f"Results will be stored in: {results_dir}")
    
    # Find the demo script
    demo_script = find_file(f"{demo_name}.py")
    if not demo_script:
        print(f"✗ {demo_name}.py not found in current directory or Code/ folder")
        return False
    
    start_time = time.time()
    
    try:
        # Change to results directory and run the demo script
        original_cwd = os.getcwd()
        
        # Set environment variable to tell demo where to save results
        env = os.environ.copy()
        env['DEMO_RESULTS_DIR'] = str(results_dir.absolute())
        
        # Run the demo script from the location where it was found
        if demo_script.startswith("Code"):
            # If script is in Code/ folder, run it from there but save results to results/
            script_dir = os.path.dirname(demo_script)
            script_name = os.path.basename(demo_script)
            
            result = subprocess.run([sys.executable, script_name], 
                                  cwd=script_dir,
                                  env=env,
                                  text=True, timeout=300)
        else:
            # Script is in current directory
            result = subprocess.run([sys.executable, demo_script], 
                                  env=env,
                                  text=True, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✓ {demo_name} completed successfully in {duration:.1f} seconds")            
            return True
        else:
            print(f"✗ {demo_name} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {demo_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ {demo_name} failed with exception: {e}")
        return False


def check_prerequisites():
    """Check if all required files are present in current directory or Code/ folder"""
    required_files = [
        "dip_hw_3.mat",
        "image_to_graph.py",
        "spectral_clustering.py", 
        "n_cuts.py",
        "demo1.py",
        "demo2.py",
        "demo3a.py",
        "demo3b.py",
        "demo3c.py"
    ]
    
    missing_files = []
    found_files = {}
    
    for file in required_files:
        found_path = find_file(file)
        if found_path:
            found_files[file] = found_path
        else:
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✓ All required files found:")
    for file, path in found_files.items():
        print(f"   - {file}")
    
    return True

def main():
    print("=== Image Segmentation Assignment - All Demos Runner ===")
    print("This script will run all five demonstration scripts in sequence.")
    print("Results will be organized in results/demo_name/ folders.")
    
    # Create main results directory
    Path(BASE_DIR, "results").mkdir(exist_ok=True)   

    # Check prerequisites
    if not check_prerequisites():
        print("\nPlease ensure all required files are in the current directory or Code/ folder.")
        return
    
    # List of demos to run
    demos = ["demo1", "demo2", "demo3a", "demo3b", "demo3c"]
    
    print(f"\nWill run {len(demos)} demonstrations:")
    for i, demo in enumerate(demos, 1):
        print(f"  {i}. {demo} - {get_demo_description(demo)}")
    
    # Confirm before running
    response = input("\nProceed with running all demos? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    # Run all demos
    start_time = time.time()
    successful_demos = []
    failed_demos = []
    
    for demo in demos:
        if run_demo(demo):
            successful_demos.append(demo)
        else:
            failed_demos.append(demo)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {total_duration:.1f} seconds")
    print(f"Successful demos: {len(successful_demos)}/{len(demos)}")
    
    if successful_demos:
        print("\n✓ Successfully completed:")
        for demo in successful_demos:
            print(f"   - {demo}")
    
    if failed_demos:
        print("\n✗ Failed demos:")
        for demo in failed_demos:
            print(f"   - {demo}")
    
    # Check output files in results directories
    print(f"\n--- Generated Output Files ---")
    total_output_files = 0
    
    for demo in successful_demos:
        demo_results_dir = Path("results") / demo
        if demo_results_dir.exists():
            output_files = list(demo_results_dir.glob("*"))
            if output_files:
                print(f"\n{demo} results ({len(output_files)} files):")
                for file in sorted(output_files):
                    print(f"   - {file}")
                total_output_files += len(output_files)
    
    if total_output_files == 0:
        print("No output files found in results directories.")
    else:
        print(f"\nTotal output files: {total_output_files}")
    
    if len(successful_demos) == len(demos):
        print(f"\n🎉 All demonstrations completed successfully!")
        print(f"Check the results/ folder for organized output files.")
    else:
        print(f"\n⚠️  Some demonstrations failed. Check error messages above.")

def get_demo_description(demo_name):
    """Get a brief description of each demo"""
    descriptions = {
        "demo1": "Spectral clustering on pre-built affinity matrix",
        "demo2": "Full pipeline (image-to-graph + spectral clustering)",
        "demo3a": "Non-recursive n-cuts vs spectral clustering",
        "demo3b": "Binary n-cuts analysis with n-cut values",
        "demo3c": "Recursive n-cuts with adaptive clustering"
    }
    return descriptions.get(demo_name, "Unknown demo")

if __name__ == "__main__":
    main()
