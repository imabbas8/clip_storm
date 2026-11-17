#!/usr/bin/env bash
# Opens a public https link to ClipStorm through Cloudflare (no account, no open ports).
# The link changes every time this starts; the current one is written to ~/clipstorm-link.txt
LINK_FILE="$HOME/clipstorm-link.txt"
echo "starting..." > "$LINK_FILE"
"$HOME/.local/bin/cloudflared" tunnel --no-autoupdate --url http://127.0.0.1:8787 2>&1 | while IFS= read -r line; do
  echo "$line"
  url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' <<<"$line")
  [ -n "$url" ] && echo "$url" > "$LINK_FILE"
done
