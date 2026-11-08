# Streamlit Web UI Layout and QA Verification Checklist


- [x] Verify Sidebar header displays `Invertis University Bareilly` and `Group: BTech_DL_041`.
- [x] Verify default model `cat_dog_classifier.keras` loads automatically at startup.
- [x] Verify custom `.keras` / `.h5` model uploader accepts user model weights.
- [x] Verify image uploader accepts `.jpg`, `.jpeg`, and `.png` files and previews image in Column 1.
- [x] Verify Column 2 displays `Predicted: CAT` (`P < 0.5`) or `Predicted: DOG` (`P >= 0.5`), confidence bar, and raw Sigmoid score.
- [x] Verify Section 6 renders live intermediate `Conv2D` feature map grids across uploaded test images.
