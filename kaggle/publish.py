"""Publishing to Kaggle: a new kit dataset version, and a kernel push — the write half of the API.

`survey.py` reads; this writes. They are split because the read half is safe to run casually and this half
is not: every call here changes something on Kaggle. Auth is shared — the same `KAGGLE_TOKEN` bearer
credential, env-only, no path baked in and nothing written to `~/.kaggle/`.

This exists because the official `kaggle` CLI authenticates with a username + key pair, which is a
different credential from the `KGAT_*` access token this project holds, so the documented
`kaggle datasets version` / `kaggle kernels push` flow cannot run here at all. The REST endpoints take the
bearer token directly.

Dataset upload is a three-step protocol: ask for an upload slot sized to the file, PUT the bytes at the URL
it returns, then create a version referencing the tokens. Kernel push is a single JSON POST carrying the
source inline.
"""

import argparse
import json
import mimetypes
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from survey import KaggleApi, MissingToken  # noqa: TID251 - sibling script, kaggle/ is on sys.path[0]

_REPO = Path(__file__).resolve().parents[1]
_BASE = "https://www.kaggle.com/api/v1"
_UPLOAD_TIMEOUT_SECONDS = 1800
_KIT = _REPO / "kaggle" / "kit"


@dataclass(frozen=True)
class Publisher:
    """Authenticated write access to the Kaggle REST API."""

    api: KaggleApi

    def _request(self, method: str, url: str, body: bytes | None, content_type: str) -> object:
        """One authenticated request, decoded from JSON when the response carries any."""
        request = urllib.request.Request(  # noqa: S310 - fixed https host
            url,
            data=body,
            method=method,
            headers={"Authorization": f"Bearer {self.api.token}", "Content-Type": content_type},
        )
        with urllib.request.urlopen(request, timeout=_UPLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
            payload = response.read().decode("utf-8", "replace")
        return json.loads(payload) if payload.strip().startswith(("{", "[")) else payload

    def post(self, path: str, body: dict[str, object]) -> object:
        """A JSON POST against the API base."""
        return self._request("POST", f"{_BASE}{path}", json.dumps(body).encode("utf-8"), "application/json")

    def upload(self, file: Path) -> str:
        """Reserve an upload slot for one file, PUT its bytes, and return the token identifying it."""
        size = file.stat().st_size
        modified = int(file.stat().st_mtime)
        slot = self.post(f"/datasets/upload/file/{size}/{modified}", {"fileName": file.name})
        if not isinstance(slot, dict) or "createUrl" not in slot:
            message = f"unexpected upload-slot response for {file.name}: {slot!r}"
            raise RuntimeError(message)
        guessed = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        self._request("PUT", str(slot["createUrl"]), file.read_bytes(), guessed)
        return str(slot.get("token", ""))


def _uploadable(directory: Path, staging: Path) -> list[Path]:
    """The dataset's top-level entries, each subdirectory zipped.

    This mirrors the documented `kaggle datasets version --dir-mode zip` flow: Kaggle stores a directory as
    an archive and restores the tree when the dataset is mounted, which is what lets the kernel glob for
    `celltrack/motion_linking.py`. Uploading the tree file-by-file would flatten it and break that lookup.
    """
    staging.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in sorted(directory.iterdir()):
        if entry.is_dir():
            archive = shutil.make_archive(str(staging / entry.name), "zip", root_dir=entry)
            entries.append(Path(archive))
        else:
            entries.append(entry)
    return entries


def _push_dataset(publisher: Publisher, arguments: argparse.Namespace) -> None:
    """Upload the kit directory and cut a new version of the dataset."""
    files = _uploadable(arguments.path, arguments.staging)
    print(f"uploading {len(files)} entries from {arguments.path}")
    tokens = []
    for file in files:
        token = publisher.upload(file)
        tokens.append(token)
        print(f"  {file.stat().st_size:>10} bytes  {file.name}")
    result = publisher.post(
        f"/datasets/createVersion/{arguments.reference.replace('/', '/')}",
        {"versionNotes": arguments.message, "files": [{"token": token} for token in tokens], "isPrivate": True},
    )
    print(json.dumps(result, indent=2) if isinstance(result, dict) else str(result))


def _create_dataset(publisher: Publisher, arguments: argparse.Namespace) -> None:
    """Create a brand-new dataset from a directory (createVersion only cuts versions of one that exists)."""
    owner, slug = arguments.reference.split("/", 1)
    files = _uploadable(arguments.path, arguments.staging)
    print(f"creating {arguments.reference} from {len(files)} entries")
    entries = []
    for file in files:
        token = publisher.upload(file)
        entries.append({"token": token, "path": file.name})
        print(f"  {file.stat().st_size:>10} bytes  {file.name}")
    result = publisher.post(
        "/datasets/create/new",
        {
            "title": arguments.title,
            "slug": slug,
            "ownerSlug": owner,
            "licenseName": "CC0-1.0",
            "isPrivate": True,
            "files": entries,
        },
    )
    print(json.dumps(result, indent=2) if isinstance(result, dict) else str(result))


def _push_kernel(publisher: Publisher, arguments: argparse.Namespace) -> None:
    """Push one kernel directory: its metadata plus the script source, inline."""
    metadata = json.loads((arguments.path / "kernel-metadata.json").read_text(encoding="utf-8"))
    source = (arguments.path / metadata["code_file"]).read_text(encoding="utf-8")
    body = {
        "id": metadata["id"],
        "title": metadata["title"],
        "text": source,
        "language": metadata["language"],
        "kernelType": metadata["kernel_type"],
        "isPrivate": metadata.get("is_private", True),
        "enableGpu": metadata.get("enable_gpu", False),
        "enableTpu": metadata.get("enable_tpu", False),
        "enableInternet": metadata.get("enable_internet", False),
        "datasetDataSources": metadata.get("dataset_sources", []),
        "competitionDataSources": metadata.get("competition_sources", []),
        "kernelDataSources": metadata.get("kernel_sources", []),
        "modelDataSources": metadata.get("model_sources", []),
        "categoryIds": [],
    }
    # The GPU type only takes effect through this field: Kaggle's default is a P100 (sm_60), which the
    # torch build in the base image refuses. See kaggle/README.md.
    if "machine_shape" in metadata:
        body["machineShape"] = metadata["machine_shape"]
    result = publisher.post("/kernels/push", body)
    print(json.dumps(result, indent=2) if isinstance(result, dict) else str(result))


def _parser() -> argparse.ArgumentParser:
    """The subcommand tree: dataset, kernel."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset", help="push a new version of a dataset from a directory")
    dataset.add_argument("--reference", default="dimiturnt/celltrack-kit")
    dataset.add_argument("--path", type=Path, default=_KIT)
    dataset.add_argument("-m", "--message", required=True, help="version notes")
    dataset.add_argument("--staging", type=Path, default=_KIT.parent / "kit_staging", help="where dir zips go")
    dataset.set_defaults(handler=_push_dataset)

    new = subparsers.add_parser("dataset-new", help="create a brand-new dataset from a directory")
    new.add_argument("--reference", required=True, help="owner/slug of the new dataset")
    new.add_argument("--title", required=True, help="human-readable dataset title")
    new.add_argument("--path", type=Path, required=True, help="directory whose entries become the dataset")
    new.add_argument("--staging", type=Path, default=_KIT.parent / "kit_staging", help="where dir zips go")
    new.set_defaults(handler=_create_dataset)

    kernel = subparsers.add_parser("kernel", help="push a kernel from its directory")
    kernel.add_argument("path", type=Path, help="kernels/<slug> directory")
    kernel.set_defaults(handler=_push_kernel)
    return parser


def main() -> None:
    """Run one publish subcommand."""
    arguments = _parser().parse_args()
    try:
        api = KaggleApi.from_environment()
    except MissingToken as error:
        raise SystemExit(str(error)) from error
    try:
        arguments.handler(Publisher(api=api), arguments)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:600]
        print(f"HTTP {error.code} {error.reason}\n{detail}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
