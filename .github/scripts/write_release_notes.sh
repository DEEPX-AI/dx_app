#!/bin/bash

set -e

# Function to get items from environment variables
get_env_items() {
    local var_name=$1
    local value="${!var_name}"
    echo "$value"
}

# Function to remove JIRA numbers from items
remove_jira_numbers() {
    echo "$1" | sed -E 's/\[[A-Za-z0-9_-]+\](\([^)]+\))?//g' | sed 's/[[:space:]]*$//'
}

# Function to update release notes section (no version bump)
update_section() {
    local file=$1
    local section_number=$2
    local items=$3
    
    if [[ -z "$items" ]]; then
        echo "DEBUG: No items for section $section_number"
        return 0
    fi
    
    echo "DEBUG: Updating section $section_number with items: [$items]"
    
    # Find the section header line
    local section_line=$(grep -n "^### ${section_number}\." "$file" | head -n 1 | cut -d: -f1)
    
    if [[ -z "$section_line" ]]; then
        echo "ERROR: Could not find section ### ${section_number}. in $file"
        return 1
    fi
    
    echo "DEBUG: Found section $section_number at line $section_line"
    
    # Determine next section line (find the next ### or ## line)
    local next_section_line
    case $section_number in
        1) 
            next_section_line=$(grep -n '^### 2\.' "$file" | head -n 1 | cut -d: -f1)
            if [[ -n "$next_section_line" ]]; then
                next_section_line=$((next_section_line - 1))
            fi
            ;;
        2) 
            next_section_line=$(grep -n '^### 3\.' "$file" | head -n 1 | cut -d: -f1)
            if [[ -n "$next_section_line" ]]; then
                next_section_line=$((next_section_line - 1))
            fi
            ;;
        3) 
            next_section_line=$(grep -n '^## v' "$file" | tail -n +2 | head -n 1 | cut -d: -f1)
            if [[ -n "$next_section_line" ]]; then
                next_section_line=$((next_section_line - 2))
            fi
            ;;
    esac
    
    # If we can't find the next section, use end of file
    if [[ -z "$next_section_line" ]]; then
        next_section_line=$(wc -l < "$file")
    fi
    
    echo "DEBUG: Next section at line $next_section_line"
    
    # Remove JIRA numbers from items
    local processed_items=$(remove_jira_numbers "$items")
    echo "DEBUG: Processed items: [$processed_items]"
    
    # Create temporary file with new content
    local temp_file="temp_section_${section_number}.md"
    
    # Build the new file content
    # 1. Content before the section
    sed -n "1,${section_line}p" "$file" > "$temp_file"
    
    # 2. Add the processed items (they should go right after the section header)
    echo "$processed_items" >> "$temp_file"
    
    # 3. Content after the current section
    if [[ $next_section_line -lt $(wc -l < "$file") ]]; then
        sed -n "$((next_section_line + 1)),\$p" "$file" >> "$temp_file"
    fi
    
    # Replace the original file
    mv "$temp_file" "$file"
    
    echo "DEBUG: Updated section $section_number successfully"
}

# Function to create new version (when bump occurred)
create_new_version() {
    local new_version=$1
    local changed_items=$2
    local fixed_items=$3
    local added_items=$4
    
    local new_version_header="## v${new_version} / $(date +'%Y-%m-%d')"
    echo "Creating new version: $new_version_header"
    
    local temp_file=$(mktemp)
    
    # Create new version content (without JIRA numbers)
    {
        echo "# RELEASE_NOTES"
        echo "$new_version_header"
        echo
        echo "### 1. Changed"
        if [[ -n "$changed_items" ]]; then
            remove_jira_numbers "$changed_items"
        fi
        echo
        echo "### 2. Fixed"
        if [[ -n "$fixed_items" ]]; then
            remove_jira_numbers "$fixed_items"
        fi
        echo
        echo "### 3. Added"
        if [[ -n "$added_items" ]]; then
            remove_jira_numbers "$added_items"
        fi
        echo
    } > "$temp_file"
    
    # Append existing content if file exists
    if [[ -f RELEASE_NOTES.md ]]; then
        local line_num=$(grep -n "^# RELEASE_NOTES$" RELEASE_NOTES.md | head -1 | cut -d: -f1)
        if [[ -n "$line_num" ]]; then
            tail -n +$((line_num + 1)) RELEASE_NOTES.md >> "$temp_file"
        else
            cat RELEASE_NOTES.md >> "$temp_file"
        fi
    fi
    
    # Move temp file to final location
    mv "$temp_file" RELEASE_NOTES.md
    
    echo "✅ Created new version release notes"
}

# Main execution
main() {
    local bump_type=$1
    local new_version=$2
    
    # Get items from environment variables
    echo "Getting items from environment variables..."
    local changed_items=$(get_env_items "CHANGED_ITEMS")
    local fixed_items=$(get_env_items "FIXED_ITEMS")
    local added_items=$(get_env_items "ADDED_ITEMS")
    
    # Debug output
    echo "Changed items: ${#changed_items} chars"
    echo "Fixed items: ${#fixed_items} chars"
    echo "Added items: ${#added_items} chars"
    
    if [[ "$bump_type" == "none" ]]; then
        echo "No version bump - updating existing release notes..."
        
        # Update RELEASE_NOTES.md (without JIRA numbers)
        if [[ -f RELEASE_NOTES.md ]]; then
            echo "Updating RELEASE_NOTES.md..."
            update_section "RELEASE_NOTES.md" "1" "$changed_items"
            update_section "RELEASE_NOTES.md" "2" "$fixed_items"
            update_section "RELEASE_NOTES.md" "3" "$added_items"
            echo "✅ Updated RELEASE_NOTES.md"
        fi
        
    else
        echo "Version bump detected - creating new version..."
        if [[ -z "$new_version" ]]; then
            echo "Error: NEW_VERSION is required when bump occurred"
            exit 1
        fi
        
        create_new_version "$new_version" "$changed_items" "$fixed_items" "$added_items"
    fi
    
    echo "🎉 Release notes update completed successfully!"
}

# Script usage
usage() {
    echo "Usage: $0 <bump_type> [new_version]"
    echo "  bump_type: 'none', 'patch', 'minor', or 'major'"
    echo "  new_version: required when bump_type is not 'none'"
    echo
    echo "Examples:"
    echo "  $0 'none'                         # Update existing version"
    echo "  $0 'minor' '1.2.0'               # Create new version 1.2.0"
    echo "  $0 'patch' '1.1.1'               # Create new version 1.1.1"
}

# Check arguments
if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

if [[ "$1" != "none" && $# -lt 2 ]]; then
    echo "Error: NEW_VERSION is required when bump occurred"
    usage
    exit 1
fi

# Run main function
main "$1" "$2"
