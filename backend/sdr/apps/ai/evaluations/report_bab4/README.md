## Bab 4 Evaluation Result Snapshot

Folder ini menyimpan snapshot artefak evaluasi yang dipakai sebagai sumber pelaporan pada [Bab_4_Evaluasi_Solusi.md](/mnt/c/life/kuliah/ta/automated-security-design-review/docs/Bab_4_Evaluasi_Solusi.md) dan [Lampiran_B_Hasil_Evaluasi_Bab_4.md](/mnt/c/life/kuliah/ta/automated-security-design-review/docs/Lampiran_B_Hasil_Evaluasi_Bab_4.md).

Struktur:

- `extraction_raw/`
  - Salinan hasil evaluasi extraction mentah yang masih tersedia di `results/extraction/`.
- `retrieval/`
  - Salinan artefak retrieval yang mendasari tabel Bab 4 saat ini.
- `debate/`
  - Salinan artefak text debate dan diagram yang dipakai pelaporan.
- `report_summaries/`
  - Ringkasan JSON yang mengikuti angka laporan final ketika laporan menggabungkan beberapa job ke dalam satu tabel/subbagian.

Catatan:

- Bab 4 dan Lampiran B sempat memuat referensi retrieval lama berbasis `design8` dan `design10`. File tersebut tidak ada lagi pada direktori hasil evaluasi aktif. Snapshot ini memakai artefak final yang benar-benar mendasari tabel Bab 4 saat ini, yaitu desain `14` dan `15`.
- Untuk evaluasi extraction, hasil mentah yang tersedia hanya untuk `job_53`, `job_56`, `job_59`, dan `job_60`. Karena laporan final juga menampilkan `job_58` dan `job_61`, folder `report_summaries/` menyimpan snapshot ringkasan JSON yang mengikuti angka laporan final agar artefak pelaporan tetap lengkap.
