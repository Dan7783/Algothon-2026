# Discord pair-review bot

This bot posts the generated four-panel pair diagnostics into one Discord
channel and presents four buttons:

- **Approve** records the pair in `discord_approved_pairs.json`.
- **Deny** records it in `discord_denied_pairs.json`.
- **Skip** defers it until the queue wraps around.
- **Stop** pauses without losing progress.

State is written atomically to
`pair_shortlist_review/discord_review_state.json`. Persistent component IDs
allow the active message to remain usable after a process restart.

## Discord setup

1. Open the [Discord Developer Portal](https://discord.com/developers/applications)
   and create an application.
2. On **Bot**, create/reset the token and keep it private.
3. On **Installation**, enable the `bot` and `applications.commands` scopes.
4. Give the bot these server permissions:
   - View Channels
   - Send Messages
   - Embed Links
   - Attach Files
   - Read Message History
   - Use Application Commands
5. Install the application into the intended server.
6. In Discord, enable Developer Mode and copy the server, review-channel, and
   reviewer user IDs.

No privileged Gateway intents are required.

## Local setup

```powershell
py -m pip install -r requirements-discord.txt

$env:DISCORD_BOT_TOKEN = "your-token"
$env:DISCORD_GUILD_ID = "your-server-id"
$env:DISCORD_CHANNEL_ID = "your-channel-id"
$env:DISCORD_REVIEWER_IDS = "your-user-id"
```

Validate all shortlist rows and images without connecting:

```powershell
py -B research\discord_pair_review_bot.py --dry-run
```

Start the bot:

```powershell
py -B research\discord_pair_review_bot.py
```

Inside the server, use:

- `/pair_review_start`
- `/pair_review_status`
- `/pair_review_stop`
- `/pair_review_export`

The slash commands are guild-scoped and should become available immediately
after the bot logs in.

## Refreshing the shortlist

Stop the bot, regenerate the shortlist, and restart:

```powershell
py -B research\shortlist_pair_review.py
py -B research\discord_pair_review_bot.py
```

If the shortlist changes, completed decisions for unchanged pair names are
preserved while the queue cursor is safely reset.
