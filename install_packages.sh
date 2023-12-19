#!/bin/bash

# File to read packages from
file="requirements.txt"

# Arrays to hold successful and failed installations
successful=()
failed=()

# Read the file line by line
while IFS= read -r package
do
    echo "Installing $package"
    pip install $package
    
    # Check the status of the pip install command
    if [ $? -eq 0 ]; then
        successful+=($package)
    else
        failed+=($package)
    fi
done < "$file"

# Print the final status
echo "Installation complete. Status:"
echo "Successful installs: ${successful[@]}"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All installations successful!"
else
    echo "Failed installs: ${failed[@]}"
fi