import boto3
import botocore.config
import json
from datetime import datetime

def blog_generation_using_bedrock(_topic:str) -> str:
    prompt = f"""Write a 200 word blog post on the topic {_topic}   
    """    
    
    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 1,
            "topP": 0.9
        }
    }
        
    try:
        bedrock = boto3.client("bedrock-runtime",region_name="us-east-1",
                               config=botocore.config.Config(read_timeout=300,retries={'max_attempts':3}))
        response = bedrock.invoke_model(
            body=json.dumps(body),
            modelId='amazon.titan-text-lite-v1',
            accept='application/json',
            contentType='application/json'
        )
        response_content = response.get('body').read()
        response_data = json.loads(response_content)
        print("Parsed response:", response_data)
        blog_details = response_data["results"][0]["outputText"]

        return blog_details
    except Exception as e:
        print('Error generating the blog!')
        print(str(e))
        return

def save_blog_details_s3(s3_key, s3_bucket, generate_blog):
    s3 = boto3.client('s3')
    
    try:
        s3.put_object(Bucket=s3_bucket, Key= s3_key, Body = generate_blog)
    
    except Exception as e:
        print('Error when saving text to S3! \n{}'.format(e))


def lambda_handler(event, context):
    event = json.loads(event["body"])
    _topic = event["blog_topic"]
    generate_blog = blog_generation_using_bedrock(_topic = _topic)
    
    if generate_blog:
        current_time = datetime.now().strftime('%H%M%S')
        s3_key = f"blog-output/{current_time}.txt"
        s3_bucket = 'aws-bedrock-test-attorri'
        save_blog_details_s3(s3_key, s3_bucket, generate_blog)
    else:
        print('No blog was generated :(')
        
    return {
        'statusCode': 200,
        'body': json.dumps('Blog Generation is completed!')
    }