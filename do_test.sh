#!/bin/sh

set -e

for name in $@; do
    echo Test name is $name
    python -m tests.$name
done
