# CRITICAL PIPELINE CONSTRAINT: OVERFITTING PREVENTION

1. **The 4-Point Trap:** Any registration yielding exactly 4 inliers with an RMSE of 0.000 px (or inliers < 15) MUST be immediately classified as an overfit matrix. Assign **Grade F**. Never report an artificial 0.000 px RMSE as successful.
2. **Spatial Distribution Check:** Before accepting a matrix, evaluate the `Spatial Entropy` and spatial coverage. If the points cover less than 15% of the crater grid (Coverage < 15%), reject the registration due to "Severe Match Clustering."
3. **Auto-Recovery:** When a low-inlier failure occurs, automatically instruct the pipeline to re-run the extraction with the `Max processing dimension (px)` upscaled (e.g., jump from 2048 to 4096) to force the extraction of higher-frequency terrain features.
