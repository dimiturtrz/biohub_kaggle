"""Surveying the competition's public state: the leaderboard, our submissions, and the public notebooks.

Where the method stands relative to the field is an input to planning, not a curiosity, and re-deriving it
by hand each session wastes the time it is meant to save. This is the read-only half of the Kaggle API
behind one entrypoint: what the pack scores, what our submissions actually scored (a kernel that ran is not
a submission that landed), and which public notebooks are new or high-signal enough to be worth reading.

Authentication is a `KAGGLE_TOKEN` environment variable holding a `KGAT_*` access token, sent as a bearer
credential. It is deliberately env-only: no token is read from a path this repo knows, written to
`~/.kaggle/`, or committed. Set it once in the shell profile and every subcommand here assumes it.

Note the competition's own quirk: the leaderboard only accepts submissions from notebooks, so `submissions`
reports what landed while `kaggle/README.md` owns the flow that puts it there.
"""

import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_COMPETITION = "biohub-cell-tracking-during-development"
_BASE = "https://www.kaggle.com/api/v1"
_TOKEN_VARIABLE = "KAGGLE_TOKEN"
_TIMEOUT_SECONDS = 90
# A model pack is hundreds of megabytes over one connection, so it gets its own budget and a chunked read.
_DOWNLOAD_TIMEOUT_SECONDS = 1800
_CHUNK_BYTES = 1 << 20
_PAGE_SIZE = 100
_MAX_PAGES = 6
# The listing is ranked three ways and merged: a notebook can be high-signal by community vote, by being
# newly run, or by Kaggle's own hotness, and no single ordering surfaces all three.
_SORTS = ("voteCount", "dateRun", "hotness")


class MissingToken(RuntimeError):
    """Raised when the environment holds no Kaggle access token."""


@dataclass(frozen=True)
class KaggleApi:
    """Read-only access to the Kaggle REST API with a bearer token taken from the environment."""

    token: str

    @classmethod
    def from_environment(cls) -> "KaggleApi":
        """Build from `KAGGLE_TOKEN`, failing loudly rather than issuing unauthenticated calls."""
        token = os.environ.get(_TOKEN_VARIABLE, "").strip()
        if not token:
            message = f"{_TOKEN_VARIABLE} is not set — export a KGAT_* access token before running this."
            raise MissingToken(message)
        return cls(token=token)

    def get(self, path: str, **parameters: str | int) -> object:
        """One authenticated GET, decoded from JSON."""
        query = urllib.parse.urlencode(parameters)
        request = urllib.request.Request(  # noqa: S310 - fixed https base, path is caller-controlled
            f"{_BASE}{path}?{query}" if query else f"{_BASE}{path}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8", "replace"))

    def download(self, path: str, destination: Path) -> int:
        """Stream one authenticated GET to a file in chunks, returning the bytes written."""
        request = urllib.request.Request(  # noqa: S310 - fixed https base, path is caller-controlled
            f"{_BASE}{path}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
            with destination.open("wb") as sink:
                return sum(sink.write(chunk) for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""))

    def leaderboard(self, competition: str) -> list[dict[str, object]]:
        """The public leaderboard, best team first."""
        payload = self.get(f"/competitions/{competition}/leaderboard/view")
        return payload["submissions"] if isinstance(payload, dict) else []

    def submissions(self, competition: str) -> list[dict[str, object]]:
        """Our own submissions to the competition, newest first."""
        payload = self.get(f"/competitions/submissions/list/{competition}")
        return payload if isinstance(payload, list) else []

    def kernels(self, competition: str) -> list[dict[str, object]]:
        """Every public notebook attached to the competition, de-duplicated across the three orderings."""
        found: dict[str, dict[str, object]] = {}
        for sort in _SORTS:
            for page in range(1, _MAX_PAGES + 1):
                rows = self.get("/kernels/list", competition=competition, sortBy=sort, pageSize=_PAGE_SIZE, page=page)
                if not isinstance(rows, list) or not rows:
                    break
                found.update({str(row["ref"]): row for row in rows})
        return list(found.values())

    def source(self, reference: str) -> str:
        """The source text of one public notebook, given its `user/slug` reference."""
        user, _, slug = reference.partition("/")
        payload = self.get("/kernels/pull", userName=user, kernelSlug=slug)
        blob = payload.get("blob", {}) if isinstance(payload, dict) else {}
        return str(blob.get("source", ""))


@dataclass(frozen=True)
class KernelRow:
    """One public notebook, reduced to the fields that decide whether it is worth reading."""

    reference: str
    title: str
    votes: int
    last_run: str

    @classmethod
    def of(cls, row: dict[str, object]) -> "KernelRow":
        """Project an API row onto the fields we rank and print."""
        return cls(
            reference=str(row["ref"]),
            title=_printable(str(row.get("title", ""))),
            votes=_as_int(row.get("totalVotes")),
            last_run=str(row.get("lastRunTime") or "")[:16],
        )

    def line(self) -> str:
        """One fixed-width row for the console listing."""
        return f"{self.votes:>4}  {self.last_run:<16}  {self.reference:<62}  {self.title[:52]}"


def _repo_relative(path: Path) -> str:
    """A path shown relative to the repo root, so console output is copy-pasteable anywhere in the tree."""
    resolved = path.resolve()
    return str(resolved.relative_to(_REPO)) if resolved.is_relative_to(_REPO) else str(resolved)


def _as_int(value: object) -> int:
    """An API count field as an int, treating a missing or non-numeric value as zero."""
    return value if isinstance(value, int) else 0


def _printable(text: str) -> str:
    """Strip the emoji and box-drawing characters a Windows console codepage cannot encode."""
    return "".join(character for character in text if character.isprintable() and ord(character) < 0x2000)


def _list_kernels(api: KaggleApi, arguments: argparse.Namespace) -> None:
    """Print the public notebooks, newest-run first when a `--since` date filters them."""
    rows = [KernelRow.of(row) for row in api.kernels(arguments.competition)]
    if arguments.since:
        rows = [row for row in rows if row.last_run >= arguments.since]
        rows.sort(key=lambda row: row.last_run, reverse=True)
    else:
        rows.sort(key=lambda row: -row.votes)
    print(f"{len(rows)} notebooks" + (f" run on/after {arguments.since}" if arguments.since else ""))
    print(f"{'votes':>4}  {'last run':<16}  {'ref':<62}  title")
    for row in rows[: arguments.limit]:
        print(row.line())


def _pull_kernels(api: KaggleApi, arguments: argparse.Namespace) -> None:
    """Download notebook sources into a directory, one file per reference."""
    arguments.out.mkdir(parents=True, exist_ok=True)
    for reference in arguments.reference:
        try:
            source = api.source(reference)
        except urllib.error.HTTPError as error:
            print(f"FAILED {reference}: HTTP {error.code}")
            continue
        destination = arguments.out / f"{reference.replace('/', '__')}.ipynb"
        destination.write_text(source, encoding="utf-8")
        print(f"{len(source):>8} bytes  {_repo_relative(destination)}")


def _safe_members(bundle: zipfile.ZipFile, target: Path) -> list[zipfile.ZipInfo]:
    """The archive entries that stay inside the target, so a crafted path cannot escape it."""
    resolved = target.resolve()
    return [member for member in bundle.infolist() if (resolved / member.filename).resolve().is_relative_to(resolved)]


def _fetch_dataset(api: KaggleApi, arguments: argparse.Namespace) -> None:
    """Download a public dataset archive and unpack it — the CC0 model packs the kernel mounts.

    Local end-to-end measurement needs the same weights the kernel gets from a Kaggle mount, and pulling
    them by hand through a browser is the sort of undocumented step that rots between sessions.
    """
    arguments.out.mkdir(parents=True, exist_ok=True)
    archive = arguments.out / f"{arguments.reference.replace('/', '__')}.zip"
    written = api.download(f"/datasets/download/{arguments.reference}", archive)
    print(f"{written} bytes -> {_repo_relative(archive)}")
    target = arguments.out / arguments.reference.partition("/")[2]
    with zipfile.ZipFile(archive) as bundle:
        members = _safe_members(bundle, target)
        skipped = len(bundle.infolist()) - len(members)
        bundle.extractall(target, members=members)  # noqa: S202 - members filtered to the target subtree
    print(
        f"extracted {len(members)} entries to {_repo_relative(target)}"
        + (f" ({skipped} unsafe skipped)" if skipped else "")
    )


def _show_leaderboard(api: KaggleApi, arguments: argparse.Namespace) -> None:
    """Print the top of the public leaderboard."""
    for rank, entry in enumerate(api.leaderboard(arguments.competition)[: arguments.limit], start=1):
        print(f"{rank:>4}  {entry.get('score', '—'):>7}  {_printable(str(entry.get('teamName', '')))[:48]}")


def _show_submissions(api: KaggleApi, arguments: argparse.Namespace) -> None:
    """Print our submissions with the score each actually landed."""
    for entry in api.submissions(arguments.competition):
        print(f"{str(entry.get('date'))[:19]}  {str(entry.get('publicScore')):>7}  {entry.get('status')}")
        print(f"          {_printable(str(entry.get('description', '')))[:150]}")


def _parser() -> argparse.ArgumentParser:
    """The subcommand tree: notebooks, pull, dataset, leaderboard, submissions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=_COMPETITION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    notebooks = subparsers.add_parser("notebooks", help="list the public notebooks")
    notebooks.add_argument("--since", default="", help="only those last run on/after this ISO date")
    notebooks.add_argument("--limit", type=int, default=40)
    notebooks.set_defaults(handler=_list_kernels)

    pull = subparsers.add_parser("pull", help="download notebook sources")
    pull.add_argument("reference", nargs="+", help="one or more user/slug references")
    pull.add_argument("--out", type=Path, default=_REPO / "scratchpad" / "nb")
    pull.set_defaults(handler=_pull_kernels)

    dataset = subparsers.add_parser("dataset", help="download and unpack a public dataset (model packs)")
    dataset.add_argument("reference", help="owner/slug of the dataset")
    dataset.add_argument("--out", type=Path, default=_REPO / "scratchpad" / "packs")
    dataset.set_defaults(handler=_fetch_dataset)

    leaderboard = subparsers.add_parser("leaderboard", help="show the public leaderboard")
    leaderboard.add_argument("--limit", type=int, default=20)
    leaderboard.set_defaults(handler=_show_leaderboard)

    submissions = subparsers.add_parser("submissions", help="show our submissions and their scores")
    submissions.set_defaults(handler=_show_submissions)
    return parser


def main() -> None:
    """Run one survey subcommand."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    arguments = _parser().parse_args()
    try:
        api = KaggleApi.from_environment()
    except MissingToken as error:
        raise SystemExit(str(error)) from error
    arguments.handler(api, arguments)


if __name__ == "__main__":
    main()
