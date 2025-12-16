import re

def extract_latex_rows(filename):
    rows = []
    with open(filename, "r") as f:
        lines = f.readlines()
    
    table_section = False
    for idx, line in enumerate(lines):
        if re.match(r'^Model(\s+\w+)+', line):  # Table header
            # Look for next non-separator, non-header line
            for l in lines[idx+2:]:
                l = l.strip()
                if l and not l.startswith("-"):
                    parts = l.split()
                    # Remove first column if it's the model name
                    values = parts[0:]
                    latex_row = " , ".join(values) #+ r" \\"
                    rows.append(latex_row)
                    break  # Only first data row
    return rows

# Usage
filename = input("Enter the filename: ").strip()
latex_rows = extract_latex_rows(filename)
for row in latex_rows:
    print(row)
