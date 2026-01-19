#!/usr/bin/env python3
"""
Artifact Migration Script

Migrates files from DataRoom/filesystem to ArtifactStore.

Usage:
    python scripts/migration/migrate_artifacts.py --source DataRoom
    python scripts/migration/migrate_artifacts.py --dry-run --source uploads
"""

import argparse
import asyncio
import hashlib
import json
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
import sys

sys.path.insert(0, ".")


class ArtifactMigrator:
    """Migrate files from filesystem to ArtifactStore."""

    def __init__(self, source_dir: str, dry_run: bool = False):
        self.source_dir = Path(source_dir)
        self.dry_run = dry_run
        self.migrated = 0
        self.skipped = 0
        self.errors = []
        self.db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://dealengine:changeme@localhost:5435/zakops"
        )

    def scan_files(self) -> List[Dict[str, Any]]:
        """Scan source directory for files."""
        files = []

        if not self.source_dir.exists():
            print(f"Source directory not found: {self.source_dir}")
            return files

        for file_path in self.source_dir.rglob("*"):
            if file_path.is_file():
                # Try to extract deal_id from path
                # Expected formats:
                #   DataRoom/{deal_id}/category/filename
                #   uploads/deals/{deal_id}/filename
                parts = file_path.relative_to(self.source_dir).parts

                deal_id = None
                category = "documents"

                # Try to find deal_id in path
                for part in parts:
                    if part.startswith("DEAL-"):
                        deal_id = part
                        break
                    # Also check for UUID format
                    try:
                        UUID(part)
                        deal_id = part
                        break
                    except ValueError:
                        continue

                # Determine category from path or filename
                if len(parts) >= 2:
                    potential_category = parts[-2].lower()
                    if potential_category in ["documents", "cim", "financials", "legal", "correspondence"]:
                        category = potential_category

                # Get file info
                mime_type, _ = mimetypes.guess_type(str(file_path))
                stat = file_path.stat()

                # Calculate SHA256
                sha256 = self._calculate_sha256(file_path)

                files.append({
                    "path": file_path,
                    "relative_path": str(file_path.relative_to(self.source_dir)),
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "mime_type": mime_type or "application/octet-stream",
                    "sha256": sha256,
                    "deal_id": deal_id,
                    "category": category,
                    "created_at": datetime.fromtimestamp(stat.st_ctime),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime)
                })

        return files

    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception:
            return ""

    async def migrate_files(self, files: List[Dict[str, Any]]) -> int:
        """Migrate files to ArtifactStore."""
        if self.dry_run:
            print(f"[DRY RUN] Would migrate {len(files)} files")
            for f in files[:5]:
                print(f"  - {f['relative_path']} -> deal: {f['deal_id']}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
            return len(files)

        import asyncpg
        conn = await asyncpg.connect(self.db_url)

        for file_info in files:
            try:
                # Skip if no deal_id (can't associate)
                if not file_info["deal_id"]:
                    print(f"  Skipping (no deal_id): {file_info['relative_path']}")
                    self.skipped += 1
                    continue

                # Check if deal exists
                deal_exists = await conn.fetchrow(
                    "SELECT deal_id FROM zakops.deals WHERE deal_id = $1",
                    file_info["deal_id"]
                )

                if not deal_exists:
                    print(f"  Skipping (deal not found): {file_info['relative_path']}")
                    self.skipped += 1
                    continue

                # Check if artifact already exists (by SHA256)
                if file_info["sha256"]:
                    existing = await conn.fetchrow(
                        "SELECT id FROM zakops.artifacts WHERE sha256 = $1 AND deal_id = $2",
                        file_info["sha256"],
                        file_info["deal_id"]
                    )
                    if existing:
                        print(f"  Skipping (duplicate): {file_info['filename']}")
                        self.skipped += 1
                        continue

                # Generate IDs
                artifact_id = uuid4()
                correlation_id = uuid4()

                # Determine storage path
                storage_backend = "local"
                storage_key = f"{file_info['deal_id']}/{file_info['category']}/{artifact_id}_{file_info['filename']}"
                storage_uri = f"file://{storage_key}"

                # Copy file to storage location
                storage_path = Path("artifacts") / storage_key
                storage_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_info["path"], "rb") as src:
                    with open(storage_path, "wb") as dst:
                        dst.write(src.read())

                # Insert artifact record
                await conn.execute(
                    """
                    INSERT INTO zakops.artifacts (
                        id, correlation_id, deal_id, filename, file_path,
                        file_type, file_size, mime_type, sha256, category,
                        metadata, storage_backend, storage_uri, storage_key,
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    """,
                    artifact_id,
                    correlation_id,
                    file_info["deal_id"],
                    file_info["filename"],
                    str(storage_path),
                    file_info["path"].suffix.lstrip(".") or None,
                    file_info["size"],
                    file_info["mime_type"],
                    file_info["sha256"] or None,
                    file_info["category"],
                    json.dumps({
                        "migrated_from": str(file_info["path"]),
                        "original_path": file_info["relative_path"],
                        "migration_date": datetime.now().isoformat()
                    }),
                    storage_backend,
                    storage_uri,
                    storage_key,
                    file_info["created_at"],
                    datetime.now()
                )

                self.migrated += 1
                print(f"  Migrated: {file_info['filename']} ({artifact_id})")

            except Exception as e:
                self.errors.append({
                    "file": file_info["relative_path"],
                    "error": str(e)
                })
                print(f"  Error: {file_info['filename']} - {e}")

        await conn.close()
        return self.migrated

    async def run(self) -> Dict[str, Any]:
        """Run the migration."""
        print(f"\nMigrating artifacts from: {self.source_dir}")
        print(f"Dry run: {self.dry_run}\n")

        if not self.source_dir.exists():
            return {
                "status": "error",
                "message": f"Source directory not found: {self.source_dir}"
            }

        # Scan
        print("Scanning files...")
        files = self.scan_files()
        print(f"  Found {len(files)} files")

        if not files:
            return {
                "status": "no_data",
                "source": str(self.source_dir),
                "files_found": 0
            }

        # Summarize by deal
        by_deal = {}
        for f in files:
            deal_id = str(f["deal_id"]) if f["deal_id"] else "unknown"
            by_deal[deal_id] = by_deal.get(deal_id, 0) + 1

        print(f"\nFiles by deal:")
        for deal_id, count in sorted(by_deal.items()):
            print(f"  {deal_id}: {count} files")

        # Migrate
        print("\nMigrating files...")
        await self.migrate_files(files)

        # Summary
        result = {
            "status": "complete",
            "source": str(self.source_dir),
            "dry_run": self.dry_run,
            "files_found": len(files),
            "files_migrated": self.migrated,
            "files_skipped": self.skipped,
            "errors": self.errors
        }

        print(f"\n{'='*40}")
        print("Migration Complete")
        print(f"{'='*40}")
        print(f"Found:    {len(files)}")
        print(f"Migrated: {self.migrated}")
        print(f"Skipped:  {self.skipped}")
        print(f"Errors:   {len(self.errors)}")

        return result


def main():
    parser = argparse.ArgumentParser(description="Migrate artifacts to ArtifactStore")
    parser.add_argument("--source", "-s", default="DataRoom", help="Source directory")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Dry run")
    parser.add_argument("--output", "-o", help="Output report file")
    args = parser.parse_args()

    migrator = ArtifactMigrator(args.source, args.dry_run)
    result = asyncio.run(migrator.run())

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
