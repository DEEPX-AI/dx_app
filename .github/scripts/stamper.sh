#!/bin/bash

# Capture the output of `git submodule status --recursive`
submodule_status=$(git submodule status --recursive)

# Temporary file to store new changelog content
temp_changelog="temp_changelog.md"

# Prepare the formatted changelog content
{
    echo "### $(date +"%Y-%m-%d")  " # Date header with two spaces for a newline in Markdown

    echo "$submodule_status" | awk '
    BEGIN {
        current_module = ""
    }

    {
        # Extract the module name (everything before the first '/')
        split($2, path_parts, "/")
        module = path_parts[1]

        # Check if the module has changed from the previous line
        if (module != current_module) {
            if (current_module != "") print ""  # Add an empty line between module sections
            printf "#### %s  \n", module  # New module header with two spaces and newline
            current_module = module
        }

        # Print the original line with two spaces at the end for Markdown newlines
        print $0 "  "
    }'
} >"$temp_changelog"

# Insert the new changelog entry below the '## Changelog' heading in the CHANGELOG.md
awk -v changelog="$temp_changelog" '
    /## Changelog/ { 
        print
        found = 1
        while ((getline line < changelog) > 0) {
            print line
        }
        next
    }
    { print }
' CHANGELOG.md >temp.md && mv temp.md CHANGELOG.md

# Clean up the temporary changelog
rm "$temp_changelog"

echo "Changelog updated successfully."
