#!/bin/bash
for item in ./*; do
    if [ -d "$item" ]; then
        echo "├── $(basename "$item")/"
        for subitem in "$item"/*; do
            if [ -e "$subitem" ]; then
                echo "│   ├── $(basename "$subitem")"
            fi
        done
    else
        echo "├── $(basename "$item")"
    fi
done
