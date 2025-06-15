from sagemaker.huggingface.model import HuggingFaceModel
# from sagemaker.huggingface import get_huggingface_model_image_uri
import sagemaker
import boto3
import json

# --- SageMaker Session and Role Setup (no change here) ---
role = "arn:aws:iam::329599621791:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"

s3_model_uri = "s3://sagemaker-us-east-1-329599621791/huggingface-qlora-mistralai-Mistral-7B--2025-06-13-22-34-50-126/output/compressed_model/model.tar.gz" # <--- **UPDATE THIS**

huggingface_model = HuggingFaceModel(
    model_data=s3_model_uri,
    role=role,
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310",
)

print("Deploying model to AWS SageMaker endpoint...")
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge", 
    container_startup_health_check_timeout=600 
)
print(f"Endpoint deployment initiated.")

# predictor.delete_endpoint() # Uncomment and run this when you're done!
