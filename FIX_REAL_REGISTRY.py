#!/usr/bin/env python3
"""Fix the real_registry.py indentation issues"""

import re

# Read the file
with open('/workspace/core/src/aura_intelligence/components/real_registry.py', 'r') as f:
    content = f.read()

# Fix the specific indentation issues in get_component_stats method
# Lines that should be indented inside the method
fixes = [
    (r'^type_counts = {}', '        type_counts = {}'),
    (r'^for component in self\.components\.values\(\):', '        for component in self.components.values():'),
    (r'^type_counts\[component\.type\.value\]', '            type_counts[component.type.value]'),
    (r'^avg_processing_time =', '        avg_processing_time ='),
    (r'^total_data_processed =', '        total_data_processed ='),
    (r'^return {', '        return {'),
    (r"^'total_components':", "            'total_components':"),
    (r"^'active_components':", "            'active_components':"),
    (r"^'type_distribution':", "            'type_distribution':"),
    (r"^'avg_processing_time_ms':", "            'avg_processing_time_ms':"),
    (r"^'total_data_processed':", "            'total_data_processed':"),
    (r"^'health_score':", "            'health_score':"),
    (r'^}$', '        }'),
    (r'^def get_components_by_type', '    def get_components_by_type'),
    (r'^"""Get components by type"""', '        """Get components by type"""'),
    (r'^# Filter by component ID', '        # Filter by component ID'),
    (r'^if component_type ==', '        if component_type =='),
    (r'^return \[c for cid', '            return [c for cid'),
    (r'^elif component_type ==', '        elif component_type =='),
    (r'^else:', '        else:'),
    (r'^def get_top_performers', '    def get_top_performers'),
    (r'^"""Get top performing components', '        """Get top performing components'),
    (r'^return sorted\(self\.components', '        return sorted(self.components'),
]

# Apply fixes
lines = content.split('\n')
for i, line in enumerate(lines):
    for pattern, replacement in fixes:
        if re.match(pattern, line):
            lines[i] = re.sub(pattern, replacement, line)
            break

# Write back
with open('/workspace/core/src/aura_intelligence/components/real_registry.py', 'w') as f:
    f.write('\n'.join(lines))

print("✅ Fixed indentation in real_registry.py")