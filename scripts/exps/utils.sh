#!/bin/bash

function wait_for_jobs_below_threshold() {
    # wait_for_jobs_below_threshold "alice" 30
    local user="${1:-zhouzhanhui}"    # Default: "zhouzhanhui" if not provided
    local max_jobs="${2:-30}"             # Default: 30 if not provided
    local current_jobs
    local sleep_interval=60       # Check every 60 seconds (adjustable)

    while true; do
        current_jobs=$(squeue -u "$user" -h | wc -l)
        if [[ "$current_jobs" -lt "$max_jobs" ]]; then
            echo "Job count for '$user' ($current_jobs) is below $max_jobs. Proceeding..."
            break
        fi
        echo "Current job count for '$user': $current_jobs (waiting for <$max_jobs)..."
        sleep "$sleep_interval"
    done
}
