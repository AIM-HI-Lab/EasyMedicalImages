import argparse
import os
import sys
import numpy as np
from EasyMedicalImages.ImagesRunner.CTImagesLibrary import CTImagesLibrary

def load_case(folder_path, image_path, seg_path, return_affine=False):
    ct = CTImagesLibrary(path_to_folder=folder_path, image_path=image_path, seg_path=seg_path, verbose=False)
    if not return_affine:
        return ct.mutated_img_data, ct.mutated_seg_data
    else:
        return ct.mutated_img_data, ct.mutated_seg_data, ct.mutated_img_affine, ct.mutated_seg_affine

def _write_volume(stream, arr):
    # Layout (must match read_volume() in HighLevel.cpp byte-for-byte):
    #   int32 ndim | int64 shape[ndim] | float32 data[prod(shape)]
    # Explicit little-endian, C-order. (x86 on both ends, so it's also native.)
    arr = np.ascontiguousarray(arr, dtype="<f4")
    stream.write(np.asarray(arr.ndim,  dtype="<i4").tobytes())
    stream.write(np.asarray(arr.shape, dtype="<i8").tobytes())
    stream.write(arr.tobytes())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--img",    required=True)
    p.add_argument("--seg",    required=True)
    p.add_argument("--return_affine", action="store_true")
    args = p.parse_args()

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

    # If load_case raises, the traceback goes to stderr and we exit non-zero with
    # nothing written to stdout, so the C++ side just skips this case (read fails
    # / close_pipe != 0). Mirrors Python's per-folder try/except.
    img, seg = load_case(args.folder, args.img, args.seg)

    _write_volume(payload, img)
    _write_volume(payload, seg)
    if args.return_affine:
        aff = np.ascontiguousarray(affine, dtype="<f8")   # 4x4 row-major float64
        sys.stdout.buffer.write(aff.tobytes())
    payload.flush()

if __name__ == "__main__":
    main()