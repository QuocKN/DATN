import numpy as np
import scipy.io
import h5py

MAT_PATH = r"C:\Users\DiepHM\Documents\Mavic_11.mat"
DAT_PATH = r"C:\Users\DiepHM\Documents\Mavic_11.dat"
VAR_NAME = "data"  # Đổi tên biến nếu file .mat dùng tên khác


def _extract_h5_value(h5_obj):
	# Group kiểu MATLAB complex thường có 2 field real/imag.
	if isinstance(h5_obj, h5py.Group):
		if "real" in h5_obj and "imag" in h5_obj:
			real = np.array(h5_obj["real"]).squeeze()
			imag = np.array(h5_obj["imag"]).squeeze()
			return real + 1j * imag
		raise ValueError(f"Unsupported HDF5 group format: {list(h5_obj.keys())}")

	data = np.array(h5_obj).squeeze()
	# Structured dtype có thể lưu real/imag dạng field.
	if data.dtype.names and "real" in data.dtype.names and "imag" in data.dtype.names:
		return data["real"] + 1j * data["imag"]
	return data


def load_mat_array(mat_path, var_name):
	try:
		mat = scipy.io.loadmat(mat_path)
		keys = [k for k in mat.keys() if not k.startswith("__")]
		if not keys:
			raise ValueError("No user variables found in MAT file.")
		print("MAT keys:", keys)
		key = var_name if var_name in mat else keys[0]
		print(f"Using variable: {key}")
		return np.array(mat[key]).squeeze()
	except NotImplementedError:
		print("Detected MATLAB v7.3 (HDF5). Falling back to h5py...")

	with h5py.File(mat_path, "r") as f:
		keys = list(f.keys())
		if not keys:
			raise ValueError("No datasets/groups found in HDF5 MAT file.")
		print("HDF5 keys:", keys)

		key = var_name if var_name in f else keys[0]
		print(f"Using variable: {key}")
		return _extract_h5_value(f[key])


def to_interleaved_iq_float32(arr):
	arr = np.asarray(arr).squeeze()

	if np.iscomplexobj(arr):
		out = np.empty(arr.size * 2, dtype=np.float32)
		out[0::2] = np.real(arr).reshape(-1).astype(np.float32)
		out[1::2] = np.imag(arr).reshape(-1).astype(np.float32)
		return out

	# Dữ liệu thực: lưu trực tiếp float32.
	return arr.reshape(-1).astype(np.float32)


def main():
	data = load_mat_array(MAT_PATH, VAR_NAME)
	data_flat = to_interleaved_iq_float32(data)
	data_flat.tofile(DAT_PATH)
	print(f"Saved DAT: {DAT_PATH}")
	print(f"Total samples written: {data_flat.size}")


if __name__ == "__main__":
	main()