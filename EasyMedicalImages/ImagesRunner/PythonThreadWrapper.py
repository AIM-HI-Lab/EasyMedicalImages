import argparse
import json
import os
import sys
import numpy as np
from EasyMedicalImages.ImagesRunner.CTImagesLibrary import CTImagesLibrary

def load_case(folder_path, image_path, seg_path, return_affine=False, verbose=False):
    ct = CTImagesLibrary(path_to_folder=folder_path, image_path=image_path, seg_path=seg_path, verbose=verbose)
    if not return_affine:
        return ct.mutated_img_data, ct.mutated_seg_data
    return ct.mutated_img_data, ct.mutated_seg_data, ct.mutated_img_affine, ct.mutated_seg_affine


# ---------------------------------------------------------------------------
# Generic per-case operation support.
#
# The C++ fan-out (HighLevel::apply_operation_to_directory) passes the name of a
# CTImagesLibrary method plus a JSON blob of keyword args. Each worker loads its
# one case, runs that method IN PLACE, then streams the resulting volumes back
# over stdout exactly like the plain load path -- so the wire protocol and the
# C++ reader are unchanged. Nothing is written to disk.
#
# Contract for an allowed operation: it must mutate ct.mutated_img_data and/or
# ct.mutated_seg_data (and their affines) in place, the way apply_seg_crop_to_image
# does. Its return value is ignored; we always re-read ct.mutated_* afterward.
# Methods that instead RETURN new arrays without mutating (e.g. the 2-D
# apply_seg_crop_to_slice) do not fit this volume-streaming path.
#
# To allow another operation, add its name here. (Delete the check entirely if you
# want to permit any CTImagesLibrary method by name.)
_ALLOWED_OPS = frozenset({
    "apply_seg_crop_to_image",
})


def _apply_operation(ct, operation, op_kwargs):
    """Run an in-place CTImagesLibrary method selected by name, with kwargs."""
    if operation not in _ALLOWED_OPS:
        raise ValueError(
            f"Operation {operation!r} is not permitted. "
            f"Allowed operations: {sorted(_ALLOWED_OPS)}"
        )
    method = getattr(ct, operation)
    method(**op_kwargs)  # mutates ct.mutated_img_data / ct.mutated_seg_data in place


def load_and_operate(folder_path, image_path, seg_path, operation="", op_kwargs=None,
                     return_affine=False, verbose=False):
    """Load one case and, if an operation is given, apply it in place before returning.

    With operation == "" this is identical to load_case (backwards compatible).
    """
    ct = CTImagesLibrary(path_to_folder=folder_path, image_path=image_path, seg_path=seg_path, verbose=verbose)
    if operation:
        _apply_operation(ct, operation, op_kwargs or {})
    if not return_affine:
        return ct.mutated_img_data, ct.mutated_seg_data
    return ct.mutated_img_data, ct.mutated_seg_data, ct.mutated_img_affine, ct.mutated_seg_affine

def _write_all(stream, buf, chunk=16 * 1024 * 1024):
    # Windows rejects a single very large write to a pipe with OSError(EINVAL,
    # errno 22). Feed the payload in <=chunk-byte slices. memoryview slicing is
    # a view, not a copy -- also avoids materializing a full arr.tobytes() copy.
    mv = memoryview(buf).cast("B")          # flat 1-D byte view (arr is C-contiguous)
    off, n = 0, mv.nbytes
    while off < n:
        off += stream.write(mv[off:off + chunk])

def _write_volume(stream, arr, affine=None):
    # int32 ndim | int64 shape[ndim] | float32 data[...] | [ float64 affine[16] ]
    arr = np.ascontiguousarray(arr, dtype="<f4")
    stream.write(np.asarray(arr.ndim,  dtype="<i4").tobytes())
    stream.write(np.asarray(arr.shape, dtype="<i8").tobytes())
    _write_all(stream, arr)                 # the (possibly huge) voxel payload
    if affine is not None:
        aff = np.ascontiguousarray(affine, dtype="<f8").reshape(4, 4)  # row-major
        stream.write(aff.tobytes())

# def _write_volume(stream, arr, affine=None):
#     # int32 ndim | int64 shape[ndim] | float32 data[...] | [ float64 affine[16] ]
#     arr = np.ascontiguousarray(arr, dtype="<f4")
#     stream.write(np.asarray(arr.ndim,  dtype="<i4").tobytes())
#     stream.write(np.asarray(arr.shape, dtype="<i8").tobytes())
#     stream.write(arr.tobytes())
#     if affine is not None:
#         aff = np.ascontiguousarray(affine, dtype="<f8").reshape(4, 4)  # row-major
#         stream.write(aff.tobytes())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--img",    required=True)
    p.add_argument("--seg",    required=True)
    p.add_argument("--return_affine", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--operation", default="",
                   help="Name of a CTImagesLibrary method to apply in place to each case "
                        "(e.g. apply_seg_crop_to_image). Empty = plain load.")
    p.add_argument("--op_kwargs", default="{}",
                   help="JSON object of keyword args for --operation.")
    args = p.parse_args()

    op_kwargs = json.loads(args.op_kwargs)

    # stdout is the data channel back to C++ (read via popen). Two invariants,
    # or the bytes get silently corrupted:
    #   1) Windows must not translate \n -> \r\n on stdout. No-op on POSIX.
    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    #   2) NOTHING but the payload may touch stdout. Grab the real binary stdout,
    #      then point text-level stdout at stderr so any stray print() -- ours,
    #      the library's, a warning -- lands on stderr instead of in the stream.
    payload = sys.stdout.buffer
    sys.stdout = sys.stderr

    if args.return_affine:
        img, seg, img_aff, seg_aff = load_and_operate(
            args.folder, args.img, args.seg,
            operation=args.operation, op_kwargs=op_kwargs,
            return_affine=True, verbose=args.verbose,
        )
        _write_volume(payload, img, img_aff)
        _write_volume(payload, seg, seg_aff)
    else:
        img, seg = load_and_operate(
            args.folder, args.img, args.seg,
            operation=args.operation, op_kwargs=op_kwargs,
            verbose=args.verbose,
        )
        _write_volume(payload, img)
        _write_volume(payload, seg)

    payload.flush()

if __name__ == "__main__":
    main()