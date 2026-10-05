# bug-inducing

Kelompok A04

Repo testing RPL

## Setup

1. Install Python 3.14 (`winget install -e --id Python.Python.3.14`).
2. Buat dan aktifkan virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Di VS Code, pilih interpreter/kernel `.venv` lalu jalankan `test.ipynb`.
   Repo target (`REPO_URL` di notebook) di-clone otomatis oleh PyDriller ke `repository/` saat run pertama, lalu dipakai ulang.