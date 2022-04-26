# Steps to track and push a model to a S3 bucket

cp models/my_best_model.h5 tracked_models/my_best_model.h5
echo "Send model to tracking directory"

dvc add tracked_models/my_best_model.h5
git add tracked_models/my_best_model.h5.dvc
git commit -m "Trained model tracked with DVC"
echo "Tracking trained model!"

# Now that the model is tracked and versioned, push the current version to S3
dvc push tracked_models/my_best_model.h5.dvc
echo "Trained model versioned and saved on the configured S3 bucket"