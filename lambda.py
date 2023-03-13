"""
Lambda Func 1: L1SerializeImageData

"""


import json
import boto3
import base64

s3 = boto3.resource('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']
    
    # Download the data from s3 to /tmp/image.png
    s3.Bucket(bucket).download_file(key, '/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

"""
Lambda Func 2: L2ImageClassification

"""

import json
#import sagemaker
import base64
import boto3
#from sagemaker.serializers import IdentitySerializer

runtime = boto3.client('runtime.sagemaker')


# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-01-18-16-06-28-591"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Instantiate a Predictor
    predictor = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType='application/x-image', Body=image)
    inferences = predictor['Body'].read().decode('utf-8')
    event["inferences"] = [float(x) for x in inferences[1:-1].split(',')]
    
    # For this model the IdentitySerializer needs to be "image/png"
    # predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    
    
    # We return the data back to the Step Function    
    
    return {
        'statusCode': 200,
        #'body': json.dumps(event)
        'body': {
            "image_data": event['image_data'],
            "s3_bucket": event['s3_bucket'],
            "s3_key": event['s3_key'],
            "inferences": event['inferences']
        }
    }


"""
Lambda Func 3: L3FilterOutLowQuality

"""

import json


THRESHOLD = .95


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event['inferences']
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any (x >= THRESHOLD for x in inferences)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': {
            "image_data": event['image_data'],
            "s3_bucket": event['s3_bucket'],
            "s3_key": event['s3_key'],
            "inferences": event['inferences']
        }
    }