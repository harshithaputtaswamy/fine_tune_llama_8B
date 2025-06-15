import boto3
import json

endpoint_name = "mistral-7B"

sm_runtime_client = boto3.client("sagemaker-runtime", region_name="us-east-1")

def wrap_with_legal_prompt(user_question: str) -> str:
    prompt = f"""
        You are a legal expert trained in U.S. immigration law. Given the following legal scenario, provide a detailed and statute-backed response.

        Scenario:
        {user_question.strip()}

        Please cite applicable provisions of the Immigration and Nationality Act (INA), the U.S. Code (8 U.S.C.), or USCIS policy manuals. Explain the legal consequences step by step, including removal eligibility, eligibility for immigration benefits, and any applicable exceptions or discretionary relief.
    """
    return prompt.strip()

payload = {
    "text": wrap_with_legal_prompt("What are the restriction for F1 visa holders"),
    "max_new_tokens": 500,
    "temperature": 0.7
}
json_payload = json.dumps(payload)

try:
    print(f"\nInvoking endpoint '{endpoint_name}' with data: {json_payload}")
    response = sm_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json", # Specify the content type of your input
        Accept="application/json",      # Specify the content type you expect in return
        Body=json_payload               # The request body (JSON string)
    )

    # Read the response body and parse it as JSON
    response_body = response["Body"].read().decode("utf-8")
    parsed_response = json.loads(response_body)

    print("\n--- Inference Response (Boto3) ---")
    print(parsed_response.get("generated_text", "No 'generated_text' found in response."))

except Exception as e:
    print(f"Error invoking endpoint: {e}")
    print("Ensure the endpoint is 'InService' and check CloudWatch logs for more details.")
