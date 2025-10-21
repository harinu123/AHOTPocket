"""Utilities for exporting HOTPocket pocket residue annotations into a Boltz-friendly
CSV schema.

This module provides a small CLI that reads a dataframe containing the standard
``pocket res`` column produced by HOTPocket. Each entry in ``pocket res`` is a
space-delimited collection of residue identifiers in the form
``[CHAIN]-[ONE LETTER AMINO ACID CODE][RESIDUE NUMBER]`` (for example,
``A-M101``). The script converts these residues into contact pairs that Boltz
expects (``[CHAIN, RESIDUE NUMBER]``) and writes them to a light-weight CSV that
can be pasted into a YAML file afterwards.

Example
-------
Assuming ``my_pockets.csv`` contains a ``pocket res`` column, running::

    python boltz_contact_export.py my_pockets.csv boltz_contacts.csv \
        --binder B --min-distance 10 --force

will create an output CSV with four columns::

    binder,contacts,min_distance,force
    B,"[[A, 126], [A, 277], [A, 37]]",10.0,true

The CSV can then be copied into a YAML document such as::

    binder: B
    contacts: [[A, 126], [A, 277], [A, 37]]
    min_distance: 10.0
    force: true

"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

POCKET_RES_COLUMN = "pocket res"
RESIDUE_PATTERN = re.compile(r"^(?P<chain>[^-]+)-(?P<aa>[A-Za-z])(?P<resid>-?\d+)$")
Row = Dict[str, str]


@dataclass(frozen=True)
class ContactResidue:
    """Parsed representation of a single residue contact."""

    chain: str
    residue_number: int

    @classmethod
    def from_token(cls, token: str) -> "ContactResidue":
        """Parse a residue token from the ``pocket res`` column.

        Parameters
        ----------
        token
            Residue identifier in the format ``CHAIN-AA123``.

        Returns
        -------
        ContactResidue
            Parsed chain identifier and residue number.

        Raises
        ------
        ValueError
            If *token* does not match the expected format.
        """

        match = RESIDUE_PATTERN.match(token.strip())
        if match is None:
            raise ValueError(
                f"Residue token '{token}' is not in the expected 'CHAIN-AA123' format."
            )
        return cls(chain=match.group("chain"), residue_number=int(match.group("resid")))

    def as_yaml_fragment(self) -> str:
        """Render a short YAML-friendly representation of the residue."""

        return f"[{self.chain}, {self.residue_number}]"


@dataclass
class ExportConfiguration:
    """Holds configuration options for a Boltz export run."""

    binder: Sequence[str]
    min_distance: Sequence[float]
    force: Sequence[bool]
    pocket_res_column: str = POCKET_RES_COLUMN
    max_contacts: int | None = None
    unique_contacts: bool = True

    def select_binder(self, index: int) -> str:
        return self.binder[index if index < len(self.binder) else -1]

    def select_min_distance(self, index: int) -> float:
        return self.min_distance[index if index < len(self.min_distance) else -1]

    def select_force(self, index: int) -> bool:
        return self.force[index if index < len(self.force) else -1]


def _ensure_sequence(value) -> List:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        if stripped.lower() in {"na", "nan", "none"}:
            return True
    return False


def _coerce_force_value(value) -> bool:
    """Convert a variety of truthy/falsey representations into a boolean."""

    if _is_missing(value):
        raise ValueError("Encountered missing value when parsing force column.")

    lowered = str(value).strip().lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(
        "Force column values must be boolean-like (true/false, yes/no, 1/0)."
    )


def _load_config(
    args: argparse.Namespace, rows: List[Row], columns: Sequence[str]
) -> ExportConfiguration:
    """Construct the export configuration based on CLI arguments."""

    if args.binder_column:
        if args.binder_column not in columns:
            raise ValueError(
                f"Binder column '{args.binder_column}' not found in input dataframe."
            )
        binder_values = [str(row[args.binder_column]) for row in rows]
    else:
        binder_values = [args.binder]

    if args.min_distance_column:
        if args.min_distance_column not in columns:
            raise ValueError(
                "Min-distance column "
                f"'{args.min_distance_column}' not found in input dataframe."
            )
        min_dist_values = []
        for row in rows:
            raw_value = row[args.min_distance_column]
            if _is_missing(raw_value):
                raise ValueError(
                    "Encountered missing value when parsing min_distance column."
                )
            try:
                min_dist_values.append(float(raw_value))
            except ValueError as exc:
                raise ValueError(
                    "Min-distance column must contain numeric values only."
                ) from exc
    else:
        min_dist_values = [args.min_distance]

    if args.force_column:
        if args.force_column not in columns:
            raise ValueError(
                f"Force column '{args.force_column}' not found in input dataframe."
            )
        force_values = [_coerce_force_value(row[args.force_column]) for row in rows]
    else:
        force_values = [args.force]

    return ExportConfiguration(
        binder=_ensure_sequence(binder_values),
        min_distance=_ensure_sequence(min_dist_values),
        force=_ensure_sequence(force_values),
        pocket_res_column=args.pocket_res_column,
        max_contacts=args.max_contacts,
        unique_contacts=not args.allow_duplicates,
    )


def parse_contacts(
    residues: Iterable[str],
    *,
    unique: bool = True,
    max_contacts: int | None = None,
) -> List[ContactResidue]:
    """Convert a collection of residue tokens into ``ContactResidue`` objects."""

    contacts: List[ContactResidue] = []
    seen: set[Tuple[str, int]] = set()

    for token in residues:
        if not token:
            continue
        contact = ContactResidue.from_token(token)
        key = (contact.chain, contact.residue_number)
        if unique and key in seen:
            continue
        contacts.append(contact)
        seen.add(key)
        if max_contacts is not None and len(contacts) >= max_contacts:
            break

    return contacts


def contacts_to_string(contacts: Sequence[ContactResidue]) -> str:
    """Render contacts as a YAML-friendly string representation."""

    if not contacts:
        return "[]"
    fragments = [contact.as_yaml_fragment() for contact in contacts]
    return "[" + ", ".join(fragments) + "]"


def force_to_string(force_value: bool) -> str:
    """Render boolean values in lowercase YAML style."""

    return "true" if bool(force_value) else "false"


def read_csv_table(path: str) -> Tuple[List[Row], Sequence[str]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV must contain a header row.")
        rows = list(reader)
    return rows, reader.fieldnames


def export_boltz_contacts(
    input_csv: str,
    output_csv: str,
    config: ExportConfiguration,
    *,
    rows: List[Row] | None = None,
    columns: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    """Export Boltz-style contacts to a CSV file.

    Parameters
    ----------
    input_csv
        Path to the HOTPocket dataframe containing a ``pocket res`` column.
    output_csv
        Path to the CSV file that should be written.
    config
        Configuration containing binder/min_distance/force overrides.
    rows, columns
        Optional pre-loaded data. If provided, *input_csv* is only used for error
        reporting.

    Returns
    -------
    list of dict
        The rows that were written to disk.
    """

    if rows is None or columns is None:
        rows, columns = read_csv_table(input_csv)

    if config.pocket_res_column not in columns:
        raise ValueError(
            f"Input dataframe must contain a '{config.pocket_res_column}' column."
        )

    output_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        pocket_res_value = row.get(config.pocket_res_column)
        if _is_missing(pocket_res_value):
            continue
        residues = str(pocket_res_value).split()
        contacts = parse_contacts(
            residues,
            unique=config.unique_contacts,
            max_contacts=config.max_contacts,
        )
        contacts_str = contacts_to_string(contacts)
        output_rows.append(
            {
                "binder": config.select_binder(idx),
                "contacts": contacts_str,
                "min_distance": config.select_min_distance(idx),
                "force": force_to_string(config.select_force(idx)),
            }
        )

    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["binder", "contacts", "min_distance", "force"]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return output_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export HOTPocket pocket residue annotations into a Boltz-friendly CSV "
            "schema."
        )
    )
    parser.add_argument("input_csv", help="Input CSV containing a 'pocket res' column")
    parser.add_argument("output_csv", help="Destination path for the exported CSV")

    binder_group = parser.add_mutually_exclusive_group()
    binder_group.add_argument(
        "--binder",
        default="B",
        help="Constant binder identifier to use for every row (default: %(default)s)",
    )
    binder_group.add_argument(
        "--binder-column",
        help="Column in the input dataframe that contains per-row binder identifiers",
    )

    min_distance_group = parser.add_mutually_exclusive_group()
    min_distance_group.add_argument(
        "--min-distance",
        type=float,
        default=10.0,
        help="Constant minimum distance to apply to every row (default: %(default)s)",
    )
    min_distance_group.add_argument(
        "--min-distance-column",
        help=(
            "Column in the input dataframe that contains per-row min_distance "
            "values"
        ),
    )

    force_group = parser.add_mutually_exclusive_group()
    force_group.add_argument(
        "--force",
        action="store_true",
        help="Force the contacts in the generated YAML",
    )
    force_group.add_argument(
        "--force-column",
        help="Column in the input dataframe that contains per-row boolean force values",
    )

    parser.add_argument(
        "--pocket-res-column",
        default=POCKET_RES_COLUMN,
        help=(
            "Name of the column that holds residue identifiers (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-contacts",
        type=int,
        help="Limit the number of contacts exported for each pocket",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Keep duplicate contacts instead of removing them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows, columns = read_csv_table(args.input_csv)
    config = _load_config(args, rows, columns)
    export_boltz_contacts(
        args.input_csv,
        args.output_csv,
        config,
        rows=rows,
        columns=columns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
