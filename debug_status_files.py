#!/usr/bin/env python3
"""
Debug script to inspect actual status files in /tmp
"""
import json
from pathlib import Path

print("=" * 60)
print("Inspecting Triton Instance Status Files")
print("=" * 60)

status_files = list(Path("/tmp").glob("triton_instance_status_*.json"))

if not status_files:
    print("\n❌ No status files found!")
    print("Expected pattern: /tmp/triton_instance_status_*.json")
else:
    print(f"\nFound {len(status_files)} status file(s):\n")
    
    total_active = 0
    max_configured = 0
    
    for status_file in sorted(status_files):
        try:
            with open(status_file, 'r') as f:
                data = json.load(f)
            
            print(f"📄 {status_file.name}")
            print(f"   active_instances: {data.get('active_instances', 'N/A')}")
            print(f"   configured_instances: {data.get('configured_instances', 'N/A')}")
            print(f"   idle_instances: {data.get('idle_instances', 'N/A')}")
            print()
            
            total_active += int(data.get('active_instances', 0))
            max_configured = max(max_configured, int(data.get('configured_instances', 0)))
            
        except Exception as e:
            print(f"❌ Error reading {status_file}: {e}\n")
    
    computed_idle = max(max_configured - total_active, 0)
    
    print("=" * 60)
    print("AGGREGATION RESULT:")
    print("=" * 60)
    print(f"Total active_instances: {total_active}")
    print(f"Max configured_instances: {max_configured}")
    print(f"Computed idle_instances: {computed_idle}")
    print()
    
    if computed_idle != (max_configured - total_active):
        print("⚠️  WARNING: Idle calculation mismatch!")
    else:
        print("✅ Aggregation logic is correct")
