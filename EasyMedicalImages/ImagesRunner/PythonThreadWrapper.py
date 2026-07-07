import argparse
import os
import sys
import numpy as np
from EasyMedicalImages.ImagesRunner.CTImagesLibrary import CTImagesLibrary

def load_case(folder_path, image_path, seg_path, return_affine=False, verbose=False):
    ct = CTImagesLibrary(path_to_folder=folder_path, image_path=image_path, seg_path=seg_path, verbose=verbose)
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

    if args.return_affine:
        img, seg, img_aff, seg_aff = load_case(args.folder, args.img, args.seg, return_affine=True, verbose=args.verbose)
        _write_volume(payload, img, img_aff)
        _write_volume(payload, seg, seg_aff)
    else:
        img, seg = load_case(args.folder, args.img, args.seg, verbose=args.verbose)
        _write_volume(payload, img)
        _write_volume(payload, seg)

    payload.flush()

if __name__ == "__main__":
    main()