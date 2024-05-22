import h5py
import argparse

def concatenate_hdf5_blockwise(src1_path, src2_path, dest_path, chunk_size=1024):
    with h5py.File(src1_path, 'r') as src1, h5py.File(src2_path, 'r') as src2, h5py.File(dest_path, 'w') as dest:
        src1_tokens = src1['tokens']
        src2_tokens = src2['tokens']
        total_size = src1_tokens.shape[0] + src2_tokens.shape[0]
        dest_tokens = dest.create_dataset('tokens', shape=(total_size,), dtype=src1_tokens.dtype)

        print(total_size)
        print(src1_tokens.shape[0], src2_tokens.shape[0])

    
        for i in range(0, src1_tokens.shape[0], chunk_size):
            end_index = min(i + chunk_size, src1_tokens.shape[0])
            dest_tokens[i:end_index] = src1_tokens[i:end_index]
            if i % 10000000:
                print(i, chunk_size, total_size)
                print(f"Finished moving chunks {i // chunk_size}/{total_size // chunk_size}")

        offset = src1_tokens.shape[0]
        for i in range(0, src2_tokens.shape[0], chunk_size):
            end_index = min(i + chunk_size, src2_tokens.shape[0])
            dest_tokens[offset + i:offset + end_index] = src2_tokens[i:end_index]
            if i % 10000000:
                print(f"Finished moving chunks {(i + offset) // chunk_size}/{total_size // chunk_size}")

        print(f"Concatenated data into {dest_path}")

def main():
    parser = argparse.ArgumentParser(description="Concatenate two HDF5 files blockwise.")
    parser.add_argument('src1_path', type=str, help='Path to the first source HDF5 file.')
    parser.add_argument('src2_path', type=str, help='Path to the second source HDF5 file.')
    parser.add_argument('dest_path', type=str, help='Path to the destination HDF5 file.')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Chunk size for blockwise copying.')
    
    args = parser.parse_args()

    concatenate_hdf5_blockwise(args.src1_path, args.src2_path, args.dest_path, args.chunk_size)

if __name__ == "__main__":
    main()
