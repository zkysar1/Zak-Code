#!/bin/sh
# Reports how many ERROR lines the application logged.
grep -c "ERROR" logs/app.log 2>/dev/null || echo 0
