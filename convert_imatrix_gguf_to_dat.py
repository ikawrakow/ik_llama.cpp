from __future__ import annotations

import os
import sys
import logging
import argparse

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf


logger = logging.getLogger("gguf-to-imatrix")


def _key_names(attr: str, fallback: str) -> set[str]:
    """Get possible GGUF key names, tolerating missing attributes."""
    names = {fallback}
    try:
        names.add(getattr(gguf.Keys.IMatrix, attr))
    except AttributeError:
        pass
    return names


CHUNK_COUNT_KEYS = _key_names('CHUNK_COUNT', 'imatrix.chunk_count')
CHUNK_SIZE_KEYS  = _key_names('CHUNK_SIZE',  'imatrix.chunk_size')
DATASET_KEYS     = _key_names('DATASETS', 'imatrix.datasets')


@dataclass
class IMatrixEntry:
    values: npt.NDArray[np.float32]
    counts: npt.NDArray[np.float32]


class IMatrixDatWriter:
    """Writes the old binary imatrix .dat format."""

    def __init__(self, outfile: Path):
        self.outfile = outfile
        self.chunk_size: int = 512
        self.chunk_count: int = 0
        self.dataset: str = ""
        self.entries: dict[str, IMatrixEntry] = {}

    def write(self) -> None:
        if self.chunk_size == 0:
            raise ValueError("chunk_size is 0, cannot write imatrix")

        with open(self.outfile, "wb") as f:
            np.array([len(self.entries)], dtype=np.int32).tofile(f)

            for name, entry in self.entries.items():
                name_bytes = name.encode("utf-8")
                np.array([len(name_bytes)], dtype=np.int32).tofile(f)
                f.write(name_bytes)

                ncall = int(entry.counts[0] / self.chunk_size)
                np.array([ncall], dtype=np.int32).tofile(f)
                np.array([len(entry.values)], dtype=np.int32).tofile(f)

                (entry.values / np.float32(self.chunk_size)).astype(np.float32).tofile(f)

                logger.debug("  %s: ncall=%d, nval=%d", name, ncall, len(entry.values))

            np.array([self.chunk_count], dtype=np.int32).tofile(f)

            dataset_bytes = self.dataset.encode("utf-8")
            np.array([len(dataset_bytes)], dtype=np.int32).tofile(f)
            if dataset_bytes:
                f.write(dataset_bytes)


class GGUFIMatrixReader:
    """Reads imatrix data from a GGUF file."""

    SUMS_SUFFIXES = (".sums", ".in_sum2")
    COUNTS_SUFFIX = ".counts"

    def __init__(self, gguf_path: Path):
        reader = gguf.GGUFReader(gguf_path)

        self.chunk_count: int = 0
        self.chunk_size: int = 512
        self.dataset: str = ""
        self.entries: dict[str, IMatrixEntry] = {}

        # --- Read KV metadata ---
        for field in reader.fields.values():
            key = field.name
            if key in CHUNK_COUNT_KEYS:
                val = int(field.parts[field.data[0]][0])
                self.chunk_count = val
            elif key in CHUNK_SIZE_KEYS:
                val = int(field.parts[field.data[0]][0])
                self.chunk_size = val
            elif key in DATASET_KEYS:
                val = bytes(field.parts[field.data[0]]).decode("utf-8")
                self.dataset = val

        # --- Read all tensors (copy + ensure float32) ---
        tensor_map: dict[str, npt.NDArray[np.float32]] = {}
        for tensor in reader.tensors:
            tensor_map[tensor.name] = np.array(tensor.data, dtype=np.float32)
            logger.debug("  Tensor: %s  shape=%s", tensor.name, tensor_map[tensor.name].shape)

        # --- Match sums/counts pairs ---
        sums_tensors:   dict[str, npt.NDArray[np.float32]] = {}
        counts_tensors: dict[str, npt.NDArray[np.float32]] = {}

        for tname, tdata in tensor_map.items():
            matched_sum = False
            for suffix in self.SUMS_SUFFIXES:
                if tname.endswith(suffix):
                    sums_tensors[tname[:-len(suffix)]] = tdata
                    matched_sum = True
                    break
            if not matched_sum and tname.endswith(self.COUNTS_SUFFIX):
                counts_tensors[tname[:-len(self.COUNTS_SUFFIX)]] = tdata

        for name, sums in sums_tensors.items():
            counts = counts_tensors.get(name)
            if counts is None:
                logger.warning("No counts tensor for %r, assuming 0", name)
                counts = np.array([0.0], dtype=np.float32)
            self.entries[name] = IMatrixEntry(values=sums, counts=counts)

        logger.info("Loaded %d imatrix entries from GGUF", len(self.entries))

        # --- Diagnostic output if nothing matched ---
        if not self.entries:
            logger.error("No imatrix tensor pairs found!")
            logger.error(
                "Expected pairs like '<name>%s' + '<name>%s'",
                self.SUMS_SUFFIXES[0], self.COUNTS_SUFFIX
            )
            if tensor_map:
                logger.error("Tensors actually present in the file (%d):", len(tensor_map))
                for n in sorted(tensor_map):
                    logger.error("  %s", n)
            else:
                logger.error("The file contains no tensors at all.")
            logger.error(
                "This file may not be a GGUF imatrix, or it may use a "
                "naming convention this script doesn't recognize yet."
            )

    def to_writer(self, outfile: Path) -> IMatrixDatWriter:
        writer = IMatrixDatWriter(outfile)
        writer.chunk_count = self.chunk_count
        writer.chunk_size = self.chunk_size
        writer.dataset = self.dataset
        writer.entries = self.entries
        return writer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a GGUF imatrix file to the old imatrix.dat format")
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "imatrix", type=Path,
        help="path to a GGUF imatrix file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.outfile is None:
        input_file: Path = args.imatrix
        if input_file.suffix == ".gguf":
            args.outfile = input_file.with_suffix(".dat")
        else:
            args.outfile = Path(str(input_file) + ".dat")

        if args.outfile.exists():
            logger.error(
                "Default output already exists, use --outfile to overwrite: %s",
                args.outfile
            )
            sys.exit(1)

    reader = GGUFIMatrixReader(args.imatrix)

    if not reader.entries:
        logger.error("Nothing to write (no entries). Re-run with --verbose for details.")
        sys.exit(1)

    writer = reader.to_writer(args.outfile)
    writer.write()

    logger.info("Wrote %d entries to %s", len(writer.entries), args.outfile)
