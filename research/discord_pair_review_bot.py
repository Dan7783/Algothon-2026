#!/usr/bin/env python
"""Discord approval queue for pair-shortlist diagnostic plots.

Required environment variables:

    DISCORD_BOT_TOKEN
    DISCORD_GUILD_ID
    DISCORD_CHANNEL_ID

Optional:

    DISCORD_REVIEWER_IDS=123456789,987654321

When reviewer IDs are omitted, members with Manage Server permission can
review pairs. The bot needs no privileged Gateway intents.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

try:
    import discord
    from discord import app_commands
    from discord.ext import commands
except ImportError as exc:  # pragma: no cover - gives a useful CLI error
    raise SystemExit(
        "discord.py is required. Install it with:\n"
        "  py -m pip install -r requirements-discord.txt"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = ROOT / "pair_shortlist_review"
STATE_VERSION = 1
Decision = Literal["approved", "denied"]
Action = Literal["approved", "denied", "skipped", "stopped"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def required_int(name: str) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        raise ValueError(f"Missing required environment variable: {name}")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer Discord ID") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive Discord ID")
    return value


def reviewer_ids_from_env() -> frozenset[int]:
    raw_value = os.environ.get("DISCORD_REVIEWER_IDS", "").strip()
    if not raw_value:
        return frozenset()
    try:
        values = frozenset(
            int(part.strip()) for part in raw_value.split(",") if part.strip()
        )
    except ValueError as exc:
        raise ValueError(
            "DISCORD_REVIEWER_IDS must be comma-separated integer user IDs"
        ) from exc
    if any(value <= 0 for value in values):
        raise ValueError("Reviewer IDs must be positive")
    return values


@dataclass(frozen=True)
class BotConfig:
    token: str
    guild_id: int
    channel_id: int
    reviewer_ids: frozenset[int]

    @classmethod
    def from_environment(cls) -> "BotConfig":
        token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
        if not token:
            raise ValueError(
                "Missing DISCORD_BOT_TOKEN. Never place the token in source."
            )
        return cls(
            token=token,
            guild_id=required_int("DISCORD_GUILD_ID"),
            channel_id=required_int("DISCORD_CHANNEL_ID"),
            reviewer_ids=reviewer_ids_from_env(),
        )


class ReviewQueue:
    """Persistent queue backed by the generated shortlist and local JSON."""

    def __init__(self, review_dir: Path) -> None:
        self.review_dir = review_dir.resolve()
        self.shortlist_path = self.review_dir / "shortlist.json"
        self.plot_dir = self.review_dir / "plots"
        self.state_path = self.review_dir / "discord_review_state.json"
        self.approved_path = self.review_dir / "discord_approved_pairs.json"
        self.denied_path = self.review_dir / "discord_denied_pairs.json"

        if not self.shortlist_path.is_file():
            raise FileNotFoundError(
                f"Missing shortlist: {self.shortlist_path}\n"
                "Run research/shortlist_pair_review.py first."
            )
        raw_shortlist = json.loads(self.shortlist_path.read_text(encoding="utf-8"))
        if not isinstance(raw_shortlist, list) or not raw_shortlist:
            raise ValueError("shortlist.json must contain at least one pair")

        self.items: list[dict[str, Any]] = []
        self.by_pair: dict[str, dict[str, Any]] = {}
        self.images: dict[str, Path] = {}
        for raw_item in raw_shortlist:
            if not isinstance(raw_item, dict):
                raise ValueError("Every shortlist item must be an object")
            pair_name = str(raw_item.get("pair_name", "")).strip()
            if not pair_name or pair_name in self.by_pair:
                raise ValueError(f"Invalid or duplicate pair_name: {pair_name!r}")
            item = dict(raw_item)
            self.items.append(item)
            self.by_pair[pair_name] = item

            safe_name = pair_name.replace("-", "_")
            candidates = sorted(self.plot_dir.glob(f"*_{safe_name}.png"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Expected one plot for {pair_name}, found {len(candidates)}"
                )
            self.images[pair_name] = candidates[0]

        signature_source = "\n".join(
            str(item["pair_name"]) for item in self.items
        ).encode("utf-8")
        self.queue_signature = hashlib.sha256(signature_source).hexdigest()
        self.state = self._load_state()
        self._write_exports()

    def _empty_state(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "queue_signature": self.queue_signature,
            "cursor": 0,
            "active_pair": None,
            "active_message_id": None,
            "decisions": {},
            "updated_at": utc_now(),
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self._empty_state()
            atomic_write_json(self.state_path, state)
            return state

        raw_state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(raw_state, dict):
            raise ValueError("discord_review_state.json must contain an object")

        decisions = raw_state.get("decisions", {})
        if not isinstance(decisions, dict):
            decisions = {}
        decisions = {
            pair_name: decision
            for pair_name, decision in decisions.items()
            if pair_name in self.by_pair
            and isinstance(decision, dict)
            and decision.get("decision") in ("approved", "denied")
        }

        signature_matches = (
            raw_state.get("queue_signature") == self.queue_signature
        )
        cursor = int(raw_state.get("cursor", 0)) if signature_matches else 0
        cursor %= len(self.items)
        active_pair = raw_state.get("active_pair") if signature_matches else None
        if active_pair not in self.by_pair or active_pair in decisions:
            active_pair = None
        active_message_id = (
            raw_state.get("active_message_id") if active_pair else None
        )

        state = self._empty_state()
        state.update(
            {
                "cursor": cursor,
                "active_pair": active_pair,
                "active_message_id": active_message_id,
                "decisions": decisions,
            }
        )
        atomic_write_json(self.state_path, state)
        return state

    @property
    def decisions(self) -> dict[str, dict[str, Any]]:
        return self.state["decisions"]

    def progress(self) -> tuple[int, int, int, int]:
        approved = sum(
            decision["decision"] == "approved"
            for decision in self.decisions.values()
        )
        denied = sum(
            decision["decision"] == "denied"
            for decision in self.decisions.values()
        )
        complete = approved + denied
        return complete, len(self.items), approved, denied

    def next_pending(self) -> tuple[int, dict[str, Any]] | None:
        if len(self.decisions) >= len(self.items):
            return None
        cursor = int(self.state["cursor"]) % len(self.items)
        for offset in range(len(self.items)):
            index = (cursor + offset) % len(self.items)
            item = self.items[index]
            if item["pair_name"] not in self.decisions:
                return index, item
        return None

    def active(self) -> tuple[dict[str, Any], int | None] | None:
        pair_name = self.state.get("active_pair")
        if pair_name not in self.by_pair:
            return None
        message_id = self.state.get("active_message_id")
        return self.by_pair[pair_name], int(message_id) if message_id else None

    def mark_active(self, pair_name: str, message_id: int) -> None:
        if pair_name not in self.by_pair:
            raise KeyError(pair_name)
        self.state["active_pair"] = pair_name
        self.state["active_message_id"] = int(message_id)
        self._save()

    def clear_active(self) -> None:
        self.state["active_pair"] = None
        self.state["active_message_id"] = None
        self._save()

    def advance_without_decision(self, pair_name: str) -> None:
        index = self._index(pair_name)
        self.state["cursor"] = (index + 1) % len(self.items)
        self.state["active_pair"] = None
        self.state["active_message_id"] = None
        self._save()

    def record_decision(
        self,
        pair_name: str,
        decision: Decision,
        *,
        user_id: int,
        user_name: str,
        message_id: int,
    ) -> None:
        if pair_name in self.decisions:
            raise ValueError(f"{pair_name} was already reviewed")
        index = self._index(pair_name)
        self.decisions[pair_name] = {
            "decision": decision,
            "user_id": int(user_id),
            "user_name": user_name,
            "message_id": int(message_id),
            "decided_at": utc_now(),
        }
        self.state["cursor"] = (index + 1) % len(self.items)
        self.state["active_pair"] = None
        self.state["active_message_id"] = None
        self._save()
        self._write_exports()

    def _index(self, pair_name: str) -> int:
        return next(
            index
            for index, item in enumerate(self.items)
            if item["pair_name"] == pair_name
        )

    def _save(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, self.state)

    def _write_exports(self) -> None:
        approved: list[dict[str, Any]] = []
        denied: list[dict[str, Any]] = []
        for item in self.items:
            pair_name = item["pair_name"]
            decision = self.decisions.get(pair_name)
            if decision is None:
                continue
            exported = {**item, "review": decision}
            if decision["decision"] == "approved":
                approved.append(exported)
            else:
                denied.append(exported)
        atomic_write_json(self.approved_path, approved)
        atomic_write_json(self.denied_path, denied)


class PairReviewService:
    def __init__(
        self,
        bot: "PairReviewBot",
        config: BotConfig,
        queue: ReviewQueue,
    ) -> None:
        self.bot = bot
        self.config = config
        self.queue = queue
        self.lock = asyncio.Lock()

    def can_review(self, interaction: discord.Interaction) -> bool:
        if interaction.guild_id != self.config.guild_id:
            return False
        if interaction.user.bot:
            return False
        if self.config.reviewer_ids:
            return interaction.user.id in self.config.reviewer_ids
        return (
            isinstance(interaction.user, discord.Member)
            and interaction.user.guild_permissions.manage_guild
        )

    async def configured_channel(
        self,
    ) -> discord.TextChannel | discord.Thread:
        channel = self.bot.get_channel(self.config.channel_id)
        if channel is None:
            channel = await self.bot.fetch_channel(self.config.channel_id)
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            raise TypeError(
                "DISCORD_CHANNEL_ID must identify a text channel or thread"
            )
        if channel.guild.id != self.config.guild_id:
            raise ValueError("Configured channel is not in DISCORD_GUILD_ID")
        return channel

    async def start_or_resume(self) -> str:
        async with self.lock:
            channel = await self.configured_channel()
            active = self.queue.active()
            if active is not None:
                item, message_id = active
                if message_id is not None:
                    try:
                        await channel.fetch_message(message_id)
                    except discord.NotFound:
                        self.queue.clear_active()
                    else:
                        return (
                            f"{item['pair_name']} is already awaiting review: "
                            f"https://discord.com/channels/{self.config.guild_id}/"
                            f"{self.config.channel_id}/{message_id}"
                        )
            message = await self._post_next_unlocked(channel)
            if message is None:
                return "Every shortlisted pair has already been reviewed."
            return (
                "Review queue started: "
                f"https://discord.com/channels/{self.config.guild_id}/"
                f"{self.config.channel_id}/{message.id}"
            )

    async def stop_queue(self) -> str:
        async with self.lock:
            active = self.queue.active()
            if active is None:
                return "The review queue is not currently active."
            _, message_id = active
            channel = await self.configured_channel()
            if message_id is not None:
                try:
                    message = await channel.fetch_message(message_id)
                    embed = self._decision_embed(
                        message, "Review paused with /pair_review_stop."
                    )
                    await message.edit(embed=embed, view=None)
                except discord.NotFound:
                    pass
            self.queue.clear_active()
            return "Review paused. Use /pair_review_start to resume this pair."

    async def handle_action(
        self,
        interaction: discord.Interaction,
        action: Action,
    ) -> None:
        async with self.lock:
            active = self.queue.active()
            if active is None:
                await interaction.response.send_message(
                    "This review message is no longer active.",
                    ephemeral=True,
                )
                return

            item, active_message_id = active
            if (
                interaction.message is None
                or interaction.message.id != active_message_id
            ):
                await interaction.response.send_message(
                    "This is an older review message. Use /pair_review_start.",
                    ephemeral=True,
                )
                return

            pair_name = str(item["pair_name"])
            if action in ("approved", "denied"):
                self.queue.record_decision(
                    pair_name,
                    action,
                    user_id=interaction.user.id,
                    user_name=str(interaction.user),
                    message_id=interaction.message.id,
                )
                summary = (
                    f"{'Approved' if action == 'approved' else 'Denied'} "
                    f"by {interaction.user}."
                )
            elif action == "skipped":
                self.queue.advance_without_decision(pair_name)
                summary = f"Skipped by {interaction.user}; it will return later."
            else:
                self.queue.clear_active()
                summary = (
                    f"Review paused by {interaction.user}. "
                    "Use /pair_review_start to resume."
                )

            embed = self._decision_embed(interaction.message, summary)
            await interaction.response.edit_message(embed=embed, view=None)

            if action == "stopped":
                return
            channel = await self.configured_channel()
            await self._post_next_unlocked(channel)

    async def _post_next_unlocked(
        self,
        channel: discord.TextChannel | discord.Thread,
    ) -> discord.Message | None:
        pending = self.queue.next_pending()
        if pending is None:
            complete, total, approved, denied = self.queue.progress()
            await channel.send(
                "✅ Pair review complete.\n"
                f"Reviewed: **{complete}/{total}** · "
                f"Approved: **{approved}** · Denied: **{denied}**"
            )
            return None

        index, item = pending
        pair_name = str(item["pair_name"])
        image_path = self.queue.images[pair_name]
        attachment_name = image_path.name
        file = discord.File(image_path, filename=attachment_name)
        embed = self._pair_embed(index, item)
        embed.set_image(url=f"attachment://{attachment_name}")
        message = await channel.send(
            content=self._progress_text(index),
            embed=embed,
            file=file,
            view=PairReviewView(self),
        )
        self.queue.mark_active(pair_name, message.id)
        return message

    def _progress_text(self, index: int) -> str:
        complete, total, approved, denied = self.queue.progress()
        return (
            f"**Pair {index + 1}/{total}** · Reviewed {complete}/{total} · "
            f"✅ {approved} · ❌ {denied}"
        )

    def _pair_embed(self, index: int, item: dict[str, Any]) -> discord.Embed:
        pair_name = str(item["pair_name"])
        embed = discord.Embed(
            title=f"Pair review #{index + 1}: {pair_name}",
            description=(
                "Approve only if the normalized prices, z-score execution, OU "
                "regime path, and P&L shape look structurally credible across "
                "all three years."
            ),
            color=discord.Color.blurple(),
        )
        embed.add_field(
            name="Robust ranking",
            value=(
                f"Robust: **{float(item['robust_score']):.3f}**\n"
                f"Full: {float(item['full_score']):.3f}\n"
                f"Worst year: {float(item['worst_year_score']):.3f}\n"
                f"Retention: {float(item['score_retention']):.1%}"
            ),
            inline=True,
        )
        embed.add_field(
            name="Year scores",
            value=(
                f"Y1: {float(item['year1_score']):.3f}\n"
                f"Y2: {float(item['year2_score']):.3f}\n"
                f"Y3: {float(item['year3_score']):.3f}\n"
                f"Min entries: {int(item['min_year_entries'])}"
            ),
            inline=True,
        )
        embed.add_field(
            name="Parameters",
            value=(
                f"`z_entry={float(item['z_entry']):.4g}`  "
                f"`z_exit={float(item['z_exit']):.4g}`\n"
                f"`q_beta={float(item['q_beta']):.4g}`  "
                f"`h={float(item['h']):.4g}`\n"
                f"`cusum_k={float(item['cusum_k']):.4g}`  "
                f"`cov_decay={float(item['cov_decay']):.4g}`"
            ),
            inline=False,
        )
        embed.set_footer(
            text="Buttons are restricted to configured reviewers or server managers."
        )
        return embed

    @staticmethod
    def _decision_embed(
        message: discord.Message,
        summary: str,
    ) -> discord.Embed:
        if message.embeds:
            embed = discord.Embed.from_dict(message.embeds[0].to_dict())
        else:
            embed = discord.Embed(title="Pair review")
        embed.color = discord.Color.dark_grey()
        embed.set_footer(text=summary)
        return embed


class PairReviewView(discord.ui.View):
    """Persistent controls; active message/pair state is resolved server-side."""

    def __init__(self, service: PairReviewService) -> None:
        super().__init__(timeout=None)
        self.service = service

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self.service.can_review(interaction):
            return True
        await interaction.response.send_message(
            "You are not authorised to review these pairs.",
            ephemeral=True,
        )
        return False

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item: discord.ui.Item[Any],
    ) -> None:
        if interaction.response.is_done():
            await interaction.followup.send(
                f"Review action failed: {error}",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                f"Review action failed: {error}",
                ephemeral=True,
            )

    @discord.ui.button(
        label="Approve",
        emoji="✅",
        style=discord.ButtonStyle.success,
        custom_id="pair-review:approve:v1",
    )
    async def approve(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button["PairReviewView"],
    ) -> None:
        await self.service.handle_action(interaction, "approved")

    @discord.ui.button(
        label="Deny",
        emoji="❌",
        style=discord.ButtonStyle.danger,
        custom_id="pair-review:deny:v1",
    )
    async def deny(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button["PairReviewView"],
    ) -> None:
        await self.service.handle_action(interaction, "denied")

    @discord.ui.button(
        label="Skip",
        emoji="⏭️",
        style=discord.ButtonStyle.secondary,
        custom_id="pair-review:skip:v1",
    )
    async def skip(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button["PairReviewView"],
    ) -> None:
        await self.service.handle_action(interaction, "skipped")

    @discord.ui.button(
        label="Stop",
        emoji="⏹️",
        style=discord.ButtonStyle.secondary,
        custom_id="pair-review:stop:v1",
    )
    async def stop(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button["PairReviewView"],
    ) -> None:
        await self.service.handle_action(interaction, "stopped")


class PairReviewBot(commands.Bot):
    def __init__(self, config: BotConfig, queue: ReviewQueue) -> None:
        super().__init__(
            command_prefix="!",
            intents=discord.Intents.default(),
            help_command=None,
        )
        self.config = config
        self.queue = queue
        self.service = PairReviewService(self, config, queue)
        install_commands(self)

    async def setup_hook(self) -> None:
        self.add_view(PairReviewView(self.service))
        guild = discord.Object(id=self.config.guild_id)
        self.tree.copy_global_to(guild=guild)
        synced = await self.tree.sync(guild=guild)
        print(
            f"Synced {len(synced)} commands to guild {self.config.guild_id}"
        )

    async def on_ready(self) -> None:
        if self.user is None:
            return
        complete, total, approved, denied = self.queue.progress()
        print(
            f"Logged in as {self.user} ({self.user.id}); "
            f"reviewed {complete}/{total}, approved {approved}, denied {denied}"
        )


def install_commands(bot: PairReviewBot) -> None:
    async def require_reviewer(interaction: discord.Interaction) -> bool:
        if bot.service.can_review(interaction):
            return True
        await interaction.response.send_message(
            "You are not authorised to control this review queue.",
            ephemeral=True,
        )
        return False

    @bot.tree.command(
        name="pair_review_start",
        description="Start or resume the pair-chart approval queue.",
    )
    async def pair_review_start(interaction: discord.Interaction) -> None:
        if not await require_reviewer(interaction):
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await bot.service.start_or_resume()
        await interaction.followup.send(result, ephemeral=True)

    @bot.tree.command(
        name="pair_review_status",
        description="Show pair-review progress and the active pair.",
    )
    async def pair_review_status(interaction: discord.Interaction) -> None:
        if not await require_reviewer(interaction):
            return
        complete, total, approved, denied = bot.queue.progress()
        active = bot.queue.active()
        active_name = active[0]["pair_name"] if active else "none"
        await interaction.response.send_message(
            f"Reviewed **{complete}/{total}** · Approved **{approved}** · "
            f"Denied **{denied}** · Active **{active_name}**",
            ephemeral=True,
        )

    @bot.tree.command(
        name="pair_review_stop",
        description="Pause the pair review without losing progress.",
    )
    async def pair_review_stop(interaction: discord.Interaction) -> None:
        if not await require_reviewer(interaction):
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await bot.service.stop_queue()
        await interaction.followup.send(result, ephemeral=True)

    @bot.tree.command(
        name="pair_review_export",
        description="Download approved and denied pair decisions.",
    )
    async def pair_review_export(interaction: discord.Interaction) -> None:
        if not await require_reviewer(interaction):
            return
        complete, total, approved, denied = bot.queue.progress()
        await interaction.response.send_message(
            f"Reviewed {complete}/{total}; approved {approved}; denied {denied}.",
            files=[
                discord.File(bot.queue.approved_path),
                discord.File(bot.queue.denied_path),
            ],
            ephemeral=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=DEFAULT_REVIEW_DIR,
        help="Directory containing shortlist.json and plots/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and print queue status without Discord login",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = ReviewQueue(args.review_dir)
    complete, total, approved, denied = queue.progress()
    print(
        f"Validated {total} pairs and {len(queue.images)} plots in "
        f"{queue.review_dir}"
    )
    print(
        f"Review state: {complete}/{total} complete; "
        f"{approved} approved; {denied} denied"
    )
    if args.dry_run:
        pending = queue.next_pending()
        if pending is not None:
            print(f"Next pair: {pending[1]['pair_name']}")
        return

    try:
        config = BotConfig.from_environment()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    bot = PairReviewBot(config, queue)
    bot.run(config.token)


if __name__ == "__main__":
    main()
